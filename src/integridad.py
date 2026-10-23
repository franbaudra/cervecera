import hashlib
from pathlib import Path
import json

def huella(ruta: Path, algoritmo: str = "sha256", bloque: int = 1 << 20) -> str:
    """Devuelve el resumen hexadecimal del archivo, leyéndolo de a 1 MiB."""
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def actualizar_checksums(rutas, carpeta_datos, archivo="checksums.json"):
    """Agrega o actualiza en checksums.json los hashes de `rutas`, sin borrar los que ya estaban.
    Las claves son rutas relativas a `carpeta_datos`, para que funcionen en cualquier máquina."""
    ruta_json = carpeta_datos / archivo
    guardados = json.load(open(ruta_json)) if ruta_json.exists() and ruta_json.stat().st_size else {}

    for r in rutas:
        guardados[r.relative_to(carpeta_datos).as_posix()] = huella(r)

    with open(ruta_json, "w") as f:
        json.dump(guardados, f, indent=2, sort_keys=True)
    return guardados

