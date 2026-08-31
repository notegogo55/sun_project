"""API สำหรับ segmentation และ tracking ของ active region"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from sunseg.data.aia import aia_to_png, region_intensities
from sunseg.inference.segment import magnetogram_to_png, parse_frame_timestamp
from sunseg.tracking.detect import detect_regions
from sunseg.tracking.tracker import build_tracker

from ..schemas import (
    AiaIntensityOut,
    DetectionOut,
    IntensitySeriesOut,
    IntensitySeriesResponse,
    LayerOut,
    SegmentResponse,
    TrackOut,
    TrackResponse,
    parse_iso,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["segmentation"])

#: คีย์ของเลเยอร์ magnetogram — เลเยอร์เดียวที่มีอยู่เสมอ
MAG_LAYER = "mag"


def _services(request: Request):
    return request.app.state.services


def _require_frames(services) -> list[str]:
    frames = services.segmentation.list_frames()
    if not frames:
        raise HTTPException(
            status_code=503,
            detail="ยังไม่มีเฟรมภาพที่ประมวลผลไว้ — "
            "รัน `python backend/scripts/download_images.py` แล้วตามด้วย `backend/scripts/build_masks.py`",
        )
    return frames


def _layer_keys(services) -> list[str]:
    """คีย์ของทุกเลเยอร์ที่ระบบรู้จัก (ไม่ว่าเฟรมไหนจะมีข้อมูลหรือไม่)"""
    return [MAG_LAYER, *(c.key for c in services.config.aia.channels)]


def _layers_for(services, timestamp: str) -> list[LayerOut]:
    """สถานะความพร้อมของทุกเลเยอร์สำหรับเฟรมนี้

    แสดงครบทุกชั้นเสมอแม้ยังไม่มีข้อมูล — ชั้นบรรยากาศสามชั้นคือสาระเชิงความรู้ของ
    หน้านี้ การซ่อนช่องที่ยังไม่ได้ดาวน์โหลดจะทำให้ผู้ใช้ไม่รู้ว่ามีอะไรให้ดูได้บ้าง
    """
    have = set(services.aia.channels_for(timestamp))
    layers = [
        LayerOut(
            key=MAG_LAYER,
            label="HMI magnetogram",
            region="โฟโตสเฟียร์ (สนามแม่เหล็ก)",
            available=True,
        )
    ]
    layers += [
        LayerOut(
            key=channel.key,
            label=channel.label,
            region=channel.region,
            available=channel.key in have,
        )
        for channel in services.config.aia.channels
    ]
    return layers


def _solar_map_for(services, timestamp: str):
    """สร้าง sunpy Map ของเฟรม เพื่อแปลงพิกเซลเป็นพิกัดเฮลิโอกราฟิก

    ใช้อาร์เรย์ศูนย์เป็นข้อมูลของ Map เพราะการแปลงพิกัด (``pixel_to_stonyhurst``) อ่านแค่
    WCS กับเวลาสังเกต ไม่แตะค่าพิกเซลเลย — จึงไม่ต้องโหลดเฟรมจริงมาก่อน และทำให้ผู้เรียก
    ทุกจุดส่ง solar map ที่เท่ากันได้เสมอ ซึ่งจำเป็นเพราะ cache ของ ``frame_masks``
    ใช้คีย์แค่ ``(timestamp, use_ground_truth)`` ถ้าสองเส้นทางส่งคนละค่าจะได้ผลไม่ตรงกัน

    ใช้ WCS **จริง** จาก ``FrameWcsStore`` ก่อนเสมอ เฟรมถูกเก็บเป็น ``.npy`` ล้วน
    (ไม่มี header) เพื่อประหยัดพื้นที่ WCS จึงถูกดึงแยกไว้ล่วงหน้าโดย
    ``download_aia.py --wcs-only``

    ถ้ายังไม่มีดัชนีนั้นจะถอยไปใช้ค่าประมาณจากค่ามาตรฐานของ HMI ซึ่งแม่นน้อยกว่า
    (รัศมีเชิงมุมแกว่ง ±1.7% ตลอดปี และจุดศูนย์กลางจานเยื้องจากกลางภาพจริงราว 12 พิกเซล)

    .. important::
       ทางถอยต้องระบุ ``rotation_angle=180°`` ด้วย — ``hmi.M_720s`` มี ``CROTA2 ≈ 180``
       เพราะเก็บภาพในทิศ CCD ดิบซึ่งกลับหัว การละเลยค่านี้เคยทำให้พิกัด lon/lat ที่คืน
       จาก ``/api/track`` ผิดทิศทั้งหมด และการชดเชยการหมุนของ tracker ดันไปผิดทาง
    """
    size = int(services.config.fulldisk.target_size)
    blank = np.zeros((size, size), dtype=np.float32)

    real_map = services.frame_wcs.solar_map(timestamp, blank)
    if real_map is not None:
        return real_map

    try:
        import astropy.units as u
        import sunpy.map
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import get_earth

        moment = parse_frame_timestamp(timestamp)
        # HMI เต็มดวงมีรัศมี ~1900 พิกเซลจาก 4096 — ปรับตามอัตราส่วนที่ย่อลงมา
        arcsec_per_pixel = 976.0 / (size * 1900.0 / 4096.0)

        reference = SkyCoord(
            0 * u.arcsec,
            0 * u.arcsec,
            obstime=moment,
            observer=get_earth(moment),
            frame="helioprojective",
        )
        header = sunpy.map.make_fitswcs_header(
            blank,
            reference,
            reference_pixel=[(size - 1) / 2, (size - 1) / 2] * u.pix,
            scale=[arcsec_per_pixel, arcsec_per_pixel] * u.arcsec / u.pix,
            rotation_angle=180 * u.deg,
            instrument="HMI",
        )
        return sunpy.map.Map(blank, header)
    except Exception as exc:  # noqa: BLE001 — ไม่มีพิกัดก็ยัง track ด้วยพิกเซลได้
        logger.debug("สร้าง solar map สำหรับ %s ไม่สำเร็จ: %s", timestamp, exc)
        return None


def _render_layer(services, timestamp: str, layer: str, magnetogram, mask, detections):
    """เรนเดอร์เลเยอร์ที่ขอ คืน ``(base64 png, คีย์เลเยอร์ที่ใช้จริง)``

    ถ้าเลเยอร์ที่ขอไม่มีข้อมูลสำหรับเฟรมนี้ จะ **ถอยกลับไป magnetogram แล้วตอบ 200**
    ไม่ใช่โยน error: คีย์ที่สะกดผิดเป็นความผิดพลาดของโปรแกรม (FastAPI ตอบ 422 ให้เอง)
    แต่ "เฟรมนี้ยังไม่ได้ดาวน์โหลด" เป็นสถานะปกติของข้อมูลที่ทยอยเติม
    """
    if layer != MAG_LAYER:
        channel = services.config.aia.channel(layer)
        image = services.aia.load(timestamp, layer) if channel else None
        if channel is not None and image is not None:
            png = aia_to_png(
                image,
                wavelength=channel.wavelength,
                vmax=channel.display_vmax,
                mask=mask,
                detections=detections,
            )
            return png, layer

    return magnetogram_to_png(magnetogram, mask=mask, detections=detections), MAG_LAYER


def _intensities_for(services, timestamp: str, detections) -> list[list[AiaIntensityOut]]:
    """ความเข้มแสงทุกช่องของทุก AR ในเฟรมนี้ (เรียงตรงกับ ``detections``)"""
    per_detection: list[list[AiaIntensityOut]] = [[] for _ in detections]
    if not detections:
        return per_detection

    for channel in services.config.aia.channels:
        image = services.aia.load(timestamp, channel.key)
        if image is None:
            continue

        for index, stats in enumerate(region_intensities(detections, image)):
            if stats["n_pixels"] == 0:
                continue
            per_detection[index].append(
                AiaIntensityOut(
                    channel=channel.key,
                    wavelength=channel.wavelength,
                    mean=round(stats["mean"], 3),
                    median=round(stats["median"], 3),
                    p95=round(stats["p95"], 3),
                    total=round(stats["total"], 3),
                    n_pixels=stats["n_pixels"],
                )
            )

    return per_detection


@router.get("/frames", response_model=list[str], summary="รายการเวลาของเฟรมที่มีอยู่")
def list_frames(request: Request, limit: int = Query(2000, ge=1, le=100000)):
    return _require_frames(_services(request))[:limit]


@router.get("/segment", response_model=SegmentResponse, summary="แบ่งส่วน AR ของเฟรมหนึ่ง")
def segment_frame(
    request: Request,
    timestamp: str = Query(..., description="เวลาของเฟรม เช่น 20141024_120000"),
    use_ground_truth: bool = Query(
        False, description="แสดง mask จริงจาก SHARP แทนผลทำนายของโมเดล"
    ),
    layer: str = Query(
        MAG_LAYER, description='เลเยอร์ภาพพื้นหลัง: "mag" หรือช่อง AIA เช่น "171"'
    ),
):
    """คืนภาพของเลเยอร์ที่เลือกพร้อมขอบ mask, รายการ AR และความเข้มแสงของแต่ละดวง

    ตั้ง ``use_ground_truth=true`` เพื่อดู mask จาก SHARP โดยตรง ซึ่งใช้ตรวจสอบ
    ข้อมูลได้แม้ยังไม่ได้เทรน U-Net

    ``layer`` เปลี่ยนแค่ **ภาพพื้นหลัง** ไม่ได้เปลี่ยนที่มาของ mask — mask มาจาก U-Net
    (หรือ SHARP) ที่มองแค่ magnetogram เสมอ ทำให้เทียบข้ามชั้นบรรยากาศได้อย่างมีความหมาย
    """
    services = _services(request)
    frames = _require_frames(services)

    if timestamp not in frames:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบเฟรมเวลา {timestamp} (มีทั้งหมด {len(frames)} เฟรม "
            f"ตั้งแต่ {frames[0]} ถึง {frames[-1]})",
        )

    if layer not in _layer_keys(services):
        raise HTTPException(
            status_code=422,
            detail=f"ไม่รู้จักเลเยอร์ {layer!r} — ที่มีให้เลือกคือ "
            f"{', '.join(_layer_keys(services))}",
        )

    try:
        magnetogram, mask, detections, dice, has_truth = services.segmentation.frame_masks(
            timestamp,
            use_ground_truth=use_ground_truth,
            min_area_px=services.tracking_config.detect.min_area_px,
            solar_map=_solar_map_for(services, timestamp),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image_png, used_layer = _render_layer(
        services, timestamp, layer, magnetogram, mask, detections
    )
    intensities = _intensities_for(services, timestamp, detections)

    return SegmentResponse(
        timestamp=timestamp,
        image_size=int(magnetogram.shape[0]),
        image_png=image_png,
        n_regions=len(detections),
        detections=[
            # detect_regions เรียงตามพื้นที่จากมากไปน้อยอยู่แล้ว rank จึงเป็นลำดับในลิสต์
            DetectionOut(
                **_detection_payload(detection, rank=index + 1),
                intensities=intensities[index],
            )
            for index, detection in enumerate(detections)
        ],
        predicted_area_fraction=round(float(mask.mean()), 6),
        has_ground_truth=has_truth,
        dice_vs_ground_truth=None if dice is None else round(dice, 4),
        layer=used_layer,
        layers=_layers_for(services, timestamp),
    )


def _series_note(used_frames: list[str], tracks: list[IntensitySeriesOut]) -> str | None:
    """คำเตือนเกี่ยวกับข้อจำกัดของช่วงเวลาที่เลือก — ``None`` ถ้าไม่มีอะไรต้องเตือน

    ความน่าเชื่อถือของเส้นขึ้นกับ **ระยะห่างระหว่างเฟรม** เป็นหลัก ไม่ใช่จำนวนเฟรม:
    ยิ่งห่าง tracker ยิ่งต้องพึ่งการทำนายตำแหน่งจากกฎการหมุนมากขึ้น และ AR เองก็อาจ
    เกิด/สลายไปทั้งดวงระหว่างช่องว่างนั้น
    """
    if len(used_frames) < 2:
        return "ช่วงนี้มีเฟรมเดียว — ขยายช่วงเวลาให้กว้างขึ้นเพื่อให้เห็นเส้นตามเวลา"

    moments = [parse_frame_timestamp(name) for name in used_frames]
    gaps = [
        (b - a).total_seconds() / 3600.0 for a, b in zip(moments, moments[1:], strict=False)
    ]
    median_gap = float(np.median(gaps))

    if tracks and all(t.n_points < 2 for t in tracks):
        return (
            f"เฟรมในช่วงนี้ห่างกัน {median_gap:.0f} ชม. ซึ่งมากเกินกว่าจะจับคู่ AR ข้ามเฟรมได้ — "
            "ลองช่วง พ.ค. 2024 ซึ่งเก็บทุก 12 ชม."
        )

    if median_gap > 48:
        return (
            f"เฟรมห่างกัน {median_gap:.0f} ชม. — การจับคู่ AR ข้ามช่องว่างขนาดนี้อาศัยการทำนาย "
            "ตำแหน่งจากกฎการหมุนเป็นหลัก จึงมีโอกาสจับคู่ผิดดวง และ AR ที่อายุสั้นกว่านี้จะหายไป "
            "ทั้งดวง · ช่วง พ.ค. 2024 เก็บทุก 12 ชม. ซึ่งติดตามได้แม่นกว่ามาก"
        )

    return None


@router.get(
    "/intensity-series",
    response_model=IntensitySeriesResponse,
    summary="ความเข้มแสงราย AR ตามเวลา แยกตามชั้นบรรยากาศ",
)
def intensity_series(
    request: Request,
    start: str = Query(..., description="เวลาเริ่มต้น เช่น 2024-05-01"),
    end: str = Query(..., description="เวลาสิ้นสุด เช่น 2024-05-31"),
    use_ground_truth: bool = Query(True, description="ใช้ mask จาก SHARP (เร็วกว่ามาก)"),
    max_frames: int = Query(120, ge=2, le=500, description="จำกัดจำนวนเฟรมเพื่อไม่ให้ตอบช้า"),
    top: int = Query(6, ge=1, le=20, description="จำนวน AR ที่คืน (เรียงตามพื้นที่สูงสุด)"),
):
    """เดินไปทีละเฟรม วัดความเข้มแสงในทุกช่อง แล้วเชื่อม AR ดวงเดียวกันข้ามเวลา

    ใช้ tracker ตัวเดียวกับ ``/api/track`` (Hungarian matching + ชดเชยการหมุนแบบ
    differential) ค่าความเข้มแสงถูกแนบไปกับ detection ก่อนป้อนเข้า tracker จึงติดตามไปกับ
    AR ดวงนั้นข้ามเฟรมได้

    .. note::
       ความหนาแน่นของเส้นขึ้นกับ cadence ของเฟรมในช่วงที่เลือก: ชุดหลัก 2011-2017 เก็บ
       ทุก 7 วัน (เห็นแนวโน้มระยะยาวได้ แต่ AR ส่วนใหญ่อายุสั้นกว่านั้นจึงจับคู่ข้ามเฟรม
       แทบไม่ได้) ส่วนหน้าต่าง case study พ.ค. 2024 เก็บทุก 12 ชม. ซึ่งติดตามได้ดี
    """
    services = _services(request)
    frames = _require_frames(services)

    start_dt = parse_iso(start, "start")
    end_dt = parse_iso(end, "end")
    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="end ต้องมาหลัง start")

    selected = [
        (name, parse_frame_timestamp(name))
        for name in frames
        if start_dt <= parse_frame_timestamp(name) <= end_dt
    ][:max_frames]

    if not selected:
        # ช่วงที่แคบกว่าระยะห่างระหว่างเฟรม (ชุดหลักเก็บทุก 7 วัน) เป็น **สถานะปกติ**
        # ของข้อมูล ไม่ใช่ข้อผิดพลาด — เกิดตั้งแต่ตอนเปิดหน้าเว็บครั้งแรกเลย เพราะช่วงเวลา
        # ถูกตั้งตามอายุของ AR ที่เลือกอัตโนมัติ ซึ่งมักสั้นกว่า 7 วัน จึงตอบ 200 พร้อม
        # คำอธิบาย แทนที่จะให้การ์ดขึ้นเป็น error สีแดง
        nearest = min(
            (parse_frame_timestamp(name) for name in frames),
            key=lambda m: min(abs((m - start_dt).days), abs((m - end_dt).days)),
        )
        nearest_date = nearest.strftime("%Y-%m-%d")
        return IntensitySeriesResponse(
            n_frames=0,
            n_tracks=0,
            frames=[],
            channels=[
                LayerOut(key=c.key, label=c.label, region=c.region, available=False)
                for c in services.config.aia.channels
            ],
            tracks=[],
            used_ground_truth=use_ground_truth,
            note=f"ไม่มีเฟรมภาพในช่วง {start} ถึง {end} — ชุดข้อมูลหลักเก็บภาพทุก 7 วัน "
            f"ลองขยายช่วงเวลาให้กว้างขึ้น (เฟรมที่ใกล้ที่สุดคือ {nearest_date})",
            nearest_frame=nearest_date,
        )

    channels = services.config.aia.channels
    tracker = build_tracker(services.tracking_config.tracker)
    detect_cfg = services.tracking_config.detect

    used_frames: list[str] = []
    channels_seen: set[str] = set()

    for name, moment in selected:
        try:
            magnetogram, _mask, detections, _dice, _has_truth = (
                services.segmentation.frame_masks(
                    name,
                    use_ground_truth=use_ground_truth,
                    min_area_px=detect_cfg.min_area_px,
                    solar_map=_solar_map_for(services, name),
                )
            )
        except (RuntimeError, FileNotFoundError):
            continue  # เฟรมที่ไม่มี mask จริงกำกับ — ข้ามไป ไม่ควรทำให้ทั้งคำขอพัง

        # แนบความเข้มแสงเข้ากับ detection **ก่อน** ป้อนเข้า tracker เพื่อให้ค่าติดตามไป
        # กับ AR ดวงนั้น หลังจับคู่แล้วจะไม่มีทางรู้ว่าค่าไหนเป็นของดวงไหนอีก
        for channel in channels:
            image = services.aia.load(name, channel.key)
            if image is None:
                continue
            channels_seen.add(channel.key)
            for detection, stats in zip(
                detections, region_intensities(detections, image), strict=True
            ):
                if stats["n_pixels"]:
                    detection.intensities[channel.key] = stats

        tracker.update(detections, moment)
        used_frames.append(name)

    tracks = sorted(
        tracker.finalise() or tracker.all_tracks,
        key=lambda t: max((p.area_px for p in t.points), default=0),
        reverse=True,
    )[:top]

    series_out = []
    for rank, track in enumerate(tracks, start=1):
        times = [p.time.isoformat() for p in track.points]
        b_peak = [round(float(p.max_abs_field), 1) for p in track.points]
        flux = [round(float(p.total_unsigned_flux), 1) for p in track.points]
        areas = [int(p.area_px) for p in track.points]
        series = {
            channel.key: [
                (
                    round(float(p.intensities[channel.key]["mean"]), 3)
                    if channel.key in p.intensities
                    else None
                )
                for p in track.points
            ]
            for channel in channels
        }
        series_out.append(
            IntensitySeriesOut(
                track_id=track.track_id,
                label=f"AR {track.track_id}" if track.track_id else f"AR {rank}",
                n_points=len(track.points),
                max_area_px=max((p.area_px for p in track.points), default=0),
                times=times,
                b_peak=b_peak,
                flux=flux,
                areas=areas,
                series=series,
            )
        )

    note = _series_note(used_frames, series_out)

    return IntensitySeriesResponse(
        n_frames=len(used_frames),
        n_tracks=len(series_out),
        frames=used_frames,
        channels=[
            LayerOut(
                key=c.key,
                label=c.label,
                region=c.region,
                available=c.key in channels_seen,
            )
            for c in channels
        ],
        tracks=series_out,
        used_ground_truth=use_ground_truth,
        note=note,
    )


@router.get("/track", response_model=TrackResponse, summary="ติดตาม AR ข้ามเวลา")
def track_regions(
    request: Request,
    start: str = Query(..., description="เวลาเริ่มต้น เช่น 2014-10-20"),
    end: str = Query(..., description="เวลาสิ้นสุด เช่น 2014-10-30"),
    use_ground_truth: bool = Query(True, description="ใช้ mask จาก SHARP (แนะนำถ้ายังไม่เทรน U-Net)"),
    max_frames: int = Query(60, ge=2, le=500, description="จำกัดจำนวนเฟรมเพื่อไม่ให้ตอบช้า"),
):
    """เดินไปทีละเฟรมตามลำดับเวลา แล้วเชื่อม AR ดวงเดียวกันเข้าด้วยกัน

    การจับคู่ชดเชยการหมุนแบบ differential ของดวงอาทิตย์ไว้แล้ว ดังนั้น AR ดวงเดียว
    จึงคงหมายเลข track เดิมตลอดหลายวันที่มันเคลื่อนข้ามจาน
    """
    services = _services(request)
    frames = _require_frames(services)

    start_dt = parse_iso(start, "start")
    end_dt = parse_iso(end, "end")
    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="end ต้องมาหลัง start")

    selected = [
        (name, parse_frame_timestamp(name))
        for name in frames
        if start_dt <= parse_frame_timestamp(name) <= end_dt
    ]
    if len(selected) < 2:
        raise HTTPException(
            status_code=404,
            detail=f"พบเฟรมเพียง {len(selected)} เฟรมในช่วงนี้ — ต้องมีอย่างน้อย 2 เฟรมจึงจะติดตามได้",
        )
    selected = selected[:max_frames]

    tracker = build_tracker(services.tracking_config.tracker)
    detect_cfg = services.tracking_config.detect

    for name, moment in selected:
        magnetogram, truth = services.segmentation.load_frame(name)

        if use_ground_truth:
            if truth is None:
                continue
            mask = truth
        else:
            if not services.segmentation.available:
                raise HTTPException(
                    status_code=503,
                    detail="ยังไม่มีโมเดล segmentation — ใช้ use_ground_truth=true แทน",
                )
            probability = services.segmentation.segment(magnetogram)
            mask = (probability >= services.segmentation.threshold).astype(np.uint8)

        detections = detect_regions(
            mask,
            magnetogram=magnetogram,
            min_area_px=detect_cfg.min_area_px,
            close_px=detect_cfg.morph_close_px,
            open_px=detect_cfg.morph_open_px,
            solar_map=_solar_map_for(services, name),
        )
        tracker.update(detections, moment)

    tracks = tracker.finalise()
    return TrackResponse(
        n_tracks=len(tracks),
        n_frames_processed=len(selected),
        tracks=[TrackOut(**t.to_dict()) for t in tracks],
    )


def _detection_payload(detection, rank: int) -> dict:
    payload = detection.to_dict()
    payload.pop("mask", None)
    payload["rank"] = rank
    return payload
