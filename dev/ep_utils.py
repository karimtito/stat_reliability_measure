import eagerpy as ep

def ep_std(x, axis=0):
    x = ep.astensor(x)
    results = (x.square().mean(axis=axis)-x.mean(axis=axis).square).sqrt()