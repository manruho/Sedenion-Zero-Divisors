import numpy as np

dim = 8

def discrete_step(v1_old, v2_old, t_old, t_new):
    v1_new = np.zeros_like(v1_old)
    v1_new[0] = np.cos(t_new)
    v1_new[1] = np.sin(t_new)

    proj = np.dot(v2_old, v1_new)
    tilde = v2_old - proj * v1_new
    n = np.linalg.norm(tilde)
    if n < 1e-12:
        tilde = np.zeros_like(v2_old)
        tilde[2] = 1.0
        tilde -= np.dot(tilde, v1_new) * v1_new
        tilde /= np.linalg.norm(tilde)
    else:
        tilde /= n
    v2_new = tilde
    return v1_new, v2_new

def run_holonomy(num_steps, init_v2_type="random", seed=0,
                 fiber_plane=(0,1)):
    t0, t1 = 0.0, 2*np.pi
    dt = (t1 - t0) / num_steps

    v1 = np.zeros(dim)
    v1[0] = 1.0

    if init_v2_type == "e2":
        v2 = np.zeros(dim); v2[2] = 1.0
    elif init_v2_type == "e3":
        v2 = np.zeros(dim); v2[3] = 1.0
    elif init_v2_type == "random":
        rng = np.random.default_rng(seed)
        v2 = rng.normal(size=dim)
        v2 -= np.dot(v2, v1) * v1
        v2 /= np.linalg.norm(v2)
    else:
        raise ValueError("unknown init_v2_type")

    thetas = []
    phis   = []
    t = t0

    i, j = fiber_plane

    for k in range(num_steps + 1):
        # base angle
        theta = np.arctan2(v1[1], v1[0])
        phi = np.arctan2(v2[j], v2[i])

        thetas.append(theta)
        phis.append(phi)

        if k < num_steps:
            t_new = t0 + (k + 1) * dt
            v1, v2 = discrete_step(v1, v2, t, t_new)
            t = t_new

    thetas = np.unwrap(np.array(thetas))
    phis   = np.unwrap(np.array(phis))

    # phi ≈ k * theta + b
    A = np.vstack([thetas, np.ones_like(thetas)]).T
    k, b = np.linalg.lstsq(A, phis, rcond=None)[0]
    return k, b

num_steps_list = [100, 200, 400, 800]
init_types = ["e2", "e3", "random"]

print("fiber angle in plane (e0, e1)")
for init in init_types:
    print(f"init_v2 = {init}")
    for n in num_steps_list:
        k, b = run_holonomy(n, init_v2_type=init, seed=42, fiber_plane=(0,1))
        print(f"  num_steps={n:4d}, k={k:.6f}")
    print()
