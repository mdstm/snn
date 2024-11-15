def update(d: dict, d1: dict):
  '''递归更新字典'''
  for k, v1 in d1.items():
    v = d.get(k)
    if isinstance(v, dict) and isinstance(v1, dict):
      update(v, v1)
    else:
      d[k] = v1
  return d
