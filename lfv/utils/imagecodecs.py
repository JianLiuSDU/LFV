def register_image_codecs() -> None:
    try:
        from imagecodecs.numcodecs import register_codecs

        register_codecs()
    except ImportError:
        print("[WARN] imagecodecs is not installed; compressed zarr arrays may fail to load.")

