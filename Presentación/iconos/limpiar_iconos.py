"""
Limpia los iconos: quita el patron de cuadros (transparencia "quemada" al
guardar como JPG) y normaliza los nombres a algo usable desde LaTeX.

Ejecutar desde la carpeta iconos:  python limpiar_iconos.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).parent

# origen -> nombre final
RENOMBRES = {
    "pidgey.jpg": "pidgey1.png",
    "pidgey 2.jpg": "pidgey2.png",
    "pidgey icon.png": "pidgey_icon.png",
    "cbea5b1a-f5ea-43a2-85e7-eb03743e63f3.jpg": "sobre.png",
    "572-5725423_delivery-truck-logo-clipart-car-van-delivery-car.png": "camion.png",
    "uber eats.png": "repartidor.png",
}

# imagenes con fondo de cuadros que hay que volver transparente
CON_CUADROS = {"pidgey1.png", "pidgey2.png", "sobre.png", "camion.png"}

TOL = 26  # tolerancia de color para considerar un pixel "fondo"


def quitar_fondo(im: Image.Image) -> Image.Image:
    """Vuelve transparente el fondo conectado al borde (cuadros o color plano)."""
    im = im.convert("RGB")
    a = np.asarray(im).astype(np.int16)
    h, w, _ = a.shape

    # colores de fondo: se muestrean en las 4 esquinas (cubre ambos tonos del damero)
    muestras = [a[0, 0], a[0, w - 1], a[h - 1, 0], a[h - 1, w - 1],
                a[0, w // 2], a[h - 1, w // 2], a[h // 2, 0], a[h // 2, w - 1]]
    colores = []
    for c in muestras:
        if not any(np.abs(np.array(c) - np.array(o)).max() <= TOL for o in colores):
            colores.append(c)

    # mascara de pixeles parecidos a algun color de fondo
    parecido = np.zeros((h, w), dtype=bool)
    for c in colores:
        parecido |= (np.abs(a - np.array(c)).max(axis=2) <= TOL)

    # solo el fondo CONECTADO al borde (no agujerea zonas claras del dibujo)
    etiquetas, _ = ndimage.label(parecido)
    del_borde = set(etiquetas[0, :]) | set(etiquetas[-1, :]) | set(etiquetas[:, 0]) | set(etiquetas[:, -1])
    del_borde.discard(0)
    fondo = np.isin(etiquetas, list(del_borde))

    rgba = np.dstack([np.asarray(im), np.where(fondo, 0, 255).astype(np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


def recortar(im: Image.Image) -> Image.Image:
    """Recorta el margen transparente sobrante."""
    bbox = im.getbbox()
    return im.crop(bbox) if bbox else im


for origen, destino in RENOMBRES.items():
    src = HERE / origen
    if not src.exists():
        print(f"[falta] {origen}")
        continue

    im = Image.open(src)
    if destino in CON_CUADROS:
        im = recortar(quitar_fondo(im))
        nota = "fondo removido + recortado"
    else:
        im = im.convert("RGBA")
        nota = "convertido a RGBA"

    im.save(HERE / destino)
    print(f"{origen:65s} -> {destino:16s} {im.size}  ({nota})")

print("\nListo.")
