"""
meter_ocr/preprocessing/perspective.py
Perspective correction / dewarping for meter display crops.
"""
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corners: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply bird's-eye view transform given 4 ordered corner points."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    max_w = max(int(wA), int(wB))

    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    max_h = max(int(hA), int(hB))

    if max_w == 0 or max_h == 0:
        return image

    dst = np.array([
        [0, 0], [max_w - 1, 0],
        [max_w - 1, max_h - 1], [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Detect the display quadrilateral and apply perspective correction.
    Falls back to mild deskew if no quad found.
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edged  = cv2.Canny(blur, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    edged  = cv2.dilate(edged, kernel, iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri  = cv2.arcLength(c, True)
        approx= cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return four_point_transform(image, approx.reshape(4, 2).astype("float32"))

    # Fallback: deskew using dominant angle
    return _deskew(image)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Simple deskew using Hough line angles."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return image
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -30 < angle < 30:
            angles.append(angle)
    if not angles:
        return image
    angle = float(np.median(angles))
    h, w  = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
