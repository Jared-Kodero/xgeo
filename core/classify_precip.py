import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0
CHUNK = 10000  # grid points per neighbour-search block (memory only)
WET_MM_H = 0.1  # wet-hour threshold (Schaer et al. 2016)
ECHO_DBZ = 10.0  # minimum composite reflectivity of an echo (Zhang et al. 2024)
CORE_DBZ = 55.0  # composite reflectivity that is always a core (Zhang et al. 2024)
PEAK_DBZ = 10.0  # peakedness amplitude a in a*cos(pi*Zbg/(2b)) (Steiner et al. 1995)
PEAK_2B = 120.0  # 2b, threshold 10*cos(pi*Zbg/120) (Zhang et al. 2024)
BKG_KM = 11.0  # background radius, grid spacing <= 1 km (Steiner et al. 1995)
BKG_PER_DX = 4.5  # background radius / spacing, 13.5 km at 3 km (Zhang et al. 2024)
CELL_KM = 7.0  # core-to-cell expansion distance, spacing >= 3 km (Zhang et al. 2024)
CELL_FINE_KM = 5.0  # same, spacing < 3 km (Zhang et al. 2024)
DIL_MIN_KM = 3.0  # minimum core dilation at 3 km spacing (Zhang et al. 2024, Eq. 1)
DIL_MAX_KM = 5.0  # maximum core dilation (Zhang et al. 2024, Eq. 1)
DIL_Z0_DBZ = 25.0  # background reflectivity where dilation starts to grow (Eq. 1)
DIL_DZ_DBZ = 5.0  # background reflectivity per 0.5 km of dilation (Eq. 1)


def _within(tree: cKDTree, xyz: np.ndarray, first: int, km: float) -> csr_matrix:
    chord = 2 * np.sin(km / (2 * R_EARTH_KM))
    w = cKDTree(xyz).sparse_distance_matrix(tree, chord, output_type="coo_matrix")
    i = np.arange(len(xyz))
    w = w.tocsr() + csr_matrix((np.ones(len(xyz)), (i, first + i)), shape=w.shape)
    w.data[:] = 1.0
    return w


def _cells(
    dbz: np.ndarray, xyz: np.ndarray, bkg_km: float, dil_min_km: float, cell_km: float
):
    x = dbz.reshape(-1, len(xyz))
    echo = x >= ECHO_DBZ
    z = np.where(echo, 10 ** (x / 10), 0.0)
    tree = cKDTree(xyz)
    chunks = [slice(i, i + CHUNK) for i in range(0, len(xyz), CHUNK)]
    core, zbg = np.zeros_like(echo), np.zeros_like(x)
    for c in chunks:
        w = _within(tree, xyz[c], c.start, bkg_km)
        zbg[:, c] = 10 * np.log10(
            np.maximum((w @ z.T).T, 1e-30) / np.maximum((w @ echo.T).T, 1)
        )
        thr = PEAK_DBZ * np.maximum(np.cos(np.pi * zbg[:, c] / PEAK_2B), 0)
        core[:, c] = echo[:, c] & ((x[:, c] - zbg[:, c] >= thr) | (x[:, c] >= CORE_DBZ))
    steps = np.ceil((zbg - DIL_Z0_DBZ) / DIL_DZ_DBZ)
    reach = np.clip(dil_min_km + 0.5 * steps, dil_min_km, DIL_MAX_KM) + cell_km
    cell = np.zeros_like(echo)
    for r in np.unique(reach[core]):
        src = core & (reach == r)
        for c in chunks:
            cell[:, c] |= (_within(tree, xyz[c], c.start, r) @ src.T).T > 0
    return (cell & echo).reshape(dbz.shape)


def classify_precip(ds: xr.Dataset) -> xr.Dataset:
    res_km = np.deg2rad(abs(float(ds.lat.diff("lat").mean()))) * R_EARTH_KM
    lat, lon = np.deg2rad(ds.lat.values)[:, None], np.deg2rad(ds.lon.values)[None, :]
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat) + 0 * lon],
        axis=-1,
    ).reshape(-1, 3)
    cells = xr.apply_ufunc(
        _cells,
        ds.max_refl,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        kwargs={
            "xyz": xyz,
            "bkg_km": max(BKG_KM, BKG_PER_DX * res_km),
            "dil_min_km": float(np.clip(2 * res_km, 1.0, DIL_MIN_KM)),
            "cell_km": CELL_FINE_KM if res_km < 3 else CELL_KM,
        },
    )
    conv_pr = ds.pr.where(cells & (ds.pr >= WET_MM_H), 0.0)
    return ds.assign(
        conv_pr=conv_pr.assign_attrs(grid_spacing_km=res_km),
        str_pr=ds.pr - conv_pr,
    )
