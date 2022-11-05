def convert_to_int(value: str, target: dict):
    for k, v in target.items():
        if str(value) == k:
            return v
