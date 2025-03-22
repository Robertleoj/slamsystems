import numba
import numpy as np

from project.utils.camera.camera_params import Intrinsics
from project.utils.camera.mvg import fundamental_matrix
from project.utils.image import to_greyscale
from project.utils.spatial.pose import Pose


@numba.njit(cache=True)
def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Generate integer pixel coordinates using Bresenham’s line algorithm"""
    pixels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        pixels.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return np.array(pixels)


@numba.njit(cache=True)
def compute_line_endpoints(
    line_homo: np.ndarray, x_low: int, x_high: int, y_low: int, y_high: int
) -> tuple[tuple[np.ndarray, np.ndarray], bool]:
    """Compute the intersection points of the line l with the given boundaries."""
    fail_ret = (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)), False

    a: float = line_homo[0]
    b: float = line_homo[1]
    c: float = line_homo[2]

    points = []

    # Intersection with left (x = x_low) and right (x = x_high)
    if abs(a) > 1e-6:  # Avoid division by zero
        y_left = int(round((-a * x_low - c) / b)) if b != 0 else y_low
        y_right = int(round((-a * x_high - c) / b)) if b != 0 else y_high
        if y_low <= y_left <= y_high:
            points.append((x_low, y_left))
        if y_low <= y_right <= y_high:
            points.append((x_high, y_right))

    # Intersection with top (y = y_low) and bottom (y = y_high)
    if abs(b) > 1e-6:
        x_top = int(round((-b * y_low - c) / a)) if a != 0 else x_low
        x_bottom = int(round((-b * y_high - c) / a)) if a != 0 else x_high
        if x_low <= x_top <= x_high:
            points.append((x_top, y_low))
        if x_low <= x_bottom <= x_high:
            points.append((x_bottom, y_high))

    if len(points) >= 2:
        return (np.array(points[0]), np.array(points[1])), True

    return fail_ret


@numba.njit(cache=True)
def try_get_block(
    img: np.ndarray, row: int, col: int, block_radius: int, height: int, width: int
) -> tuple[np.ndarray, bool]:
    fail_ret = (np.zeros((0, 0)), False)

    row_low = row - block_radius

    if row_low < 0:
        return fail_ret

    row_high = row + block_radius
    if row_high >= height:
        return fail_ret

    col_low = col - block_radius
    if col_low < 0:
        return fail_ret

    col_high = col + block_radius
    if col_high >= width:
        return fail_ret

    block = img[row_low : row_high + 1, col_low : col_high + 1]

    block = block - np.mean(block)

    return block, True


@numba.njit(cache=True)
def block_loss_SSD(block1: np.ndarray, block2: np.ndarray) -> float:
    return np.square(block1 - block2).sum()


@numba.njit(cache=True)
def block_score_NCC(block1: np.ndarray, block2: np.ndarray) -> float:
    num = np.sum(block1 * block2)
    sum1 = np.square(block1).sum()
    sum2 = np.square(block2).sum()
    denom = np.sqrt(sum1 * sum2)

    if denom < 1e-6:
        return 0.0
    return num / denom


@numba.njit(cache=True)
def epipolar_line_search(
    grey1: np.ndarray,
    grey2: np.ndarray,
    row: int,
    col: int,
    block_radius: int,
    fundamental_matrix: np.ndarray,
    search_zone_size: int,
) -> tuple[np.ndarray, bool]:
    fail_ret = (np.zeros((0,), dtype=np.int64), False)

    height = grey1.shape[0]
    width = grey1.shape[1]

    im1_block, success = try_get_block(grey1, row, col, block_radius, height, width)

    if not success:
        return fail_ret

    x = col
    y = row

    p1 = np.array([x, y, 1]).reshape(3, 1)
    line = (fundamental_matrix @ p1.astype(np.float64)).flatten()

    search_radius = search_zone_size // 2
    endpoints, success = compute_line_endpoints(
        line,
        max(col - search_radius, 0),
        min(col + search_radius + 1, width),
        max(row - search_radius, 0),
        min(row + search_radius + 1, height),
    )

    if not success:
        return fail_ret

    endpoint1 = endpoints[0]
    endpoint2 = endpoints[1]

    line_points = bresenham_line(endpoint1[0], endpoint1[1], endpoint2[0], endpoint2[1])

    found = False
    best_score = -1.0
    best_pixel: np.ndarray = fail_ret[0]

    for pixel in line_points:
        x = pixel[0]
        y = pixel[1]
        im2_block, success = try_get_block(grey2, y, x, block_radius, height, width)

        if not success:
            continue

        score = block_score_NCC(im1_block, im2_block)

        if score > best_score and score > 0.90:
            best_score = score
            best_pixel = pixel
            found = True

    return best_pixel, found


@numba.njit(parallel=True, cache=True)
def block_search_numba(
    pixels: np.ndarray,
    grey1: np.ndarray,
    grey2: np.ndarray,
    fundamental_matrix: np.ndarray,
    block_size: int,
    search_zone_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    N = pixels.shape[0]

    block_radius = block_size // 2

    valid_mask = np.ones((N,), dtype=np.uint8)

    matched_pixels = np.zeros((N, 2))

    # loop through all pixels in grey1 to get a depth estimate for it

    for i in numba.prange(N):
        pixel = pixels[i]
        col = pixel[0]
        row = pixel[1]

        matched_pixel, success = epipolar_line_search(
            grey1, grey2, row, col, block_radius, fundamental_matrix, search_zone_size
        )

        if not success:
            valid_mask[i] = 0
        else:
            matched_pixels[i] = matched_pixel

    return matched_pixels, valid_mask


def epipolar_line_search_block_match(
    pixels: np.ndarray,
    im1: np.ndarray,
    im2: np.ndarray,
    cam1_to_cam2: Pose,
    intrinsics: Intrinsics,
    block_size: int = 9,
    undistort: bool = False,
    search_zone_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if undistort:
        im1 = intrinsics.undistort_image(im1)
        im2 = intrinsics.undistort_image(im2)

    F = fundamental_matrix(cam1_to_cam2, intrinsics.camera_matrix)
    grey1 = to_greyscale(im1)
    grey2 = to_greyscale(im2)

    matched_pixels, valid_mask = block_search_numba(pixels, grey1, grey2, F, block_size, search_zone_size)

    return matched_pixels, valid_mask
