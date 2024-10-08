import numpy as np
from matplotlib import pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# -- Vertices
train_electron_x = np.load("train_electron_x.npy")
train_muon_x = np.load("train_muon_x.npy")
train_pion_x = np.load("train_pion_x.npy")
train_proton_x = np.load("train_proton_x.npy")
test_electron_x = np.load("test_electron_x.npy")
test_muon_x = np.load("test_muon_x.npy")
test_pion_x = np.load("test_pion_x.npy")
test_proton_x = np.load("test_proton_x.npy")
print(train_electron_x)
print(train_muon_x)
print(train_pion_x)
print(train_proton_x)
print(len(train_electron_x), ",",  len(np.unique(train_electron_x)))
print(len(train_muon_x), ",",  len(np.unique(train_muon_x)))
print(len(train_pion_x), ",",  len(np.unique(train_pion_x)))
print(len(train_proton_x), ",",  len(np.unique(train_proton_x)))
print("----")
print(len(test_electron_x), ",",  len(np.unique(test_electron_x)))
print(len(test_muon_x), ",",  len(np.unique(test_muon_x)))
print(len(test_pion_x), ",",  len(np.unique(test_pion_x)))
print(len(test_proton_x), ",",  len(np.unique(test_proton_x)))
print("----")
print(
    len(
        np.unique(
            np.concatenate(
                [
                    train_electron_x, test_electron_x,
                    train_muon_x, test_muon_x,
                    train_pion_x, test_pion_x,
                    train_proton_x, test_proton_x
                ]
            )
        )
    )
)
print()
plt.hist(
    train_electron_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[0],
    label="electron",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_electron_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[0],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_pion_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[1],
    label="pion",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_pion_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[1],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_muon_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[2],
    label="muon",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_muon_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[2],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_proton_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[3],
    label="proton",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_proton_x,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[3],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.legend()
plt.ylim(0, 0.001)
plt.savefig("vtx_x.pdf")
plt.close()

train_electron_y = np.load("train_electron_y.npy")
train_muon_y = np.load("train_muon_y.npy")
train_pion_y = np.load("train_pion_y.npy")
train_proton_y = np.load("train_proton_y.npy")
test_electron_y = np.load("test_electron_y.npy")
test_muon_y = np.load("test_muon_y.npy")
test_pion_y = np.load("test_pion_y.npy")
test_proton_y = np.load("test_proton_y.npy")
print(len(train_electron_y), ",",  len(np.unique(train_electron_y)))
print(len(train_muon_y), ",",  len(np.unique(train_muon_y)))
print(len(train_pion_y), ",",  len(np.unique(train_pion_y)))
print(len(train_proton_y), ",",  len(np.unique(train_proton_y)))
print("----")
print(len(test_electron_y), ",",  len(np.unique(test_electron_y)))
print(len(test_muon_y), ",",  len(np.unique(test_muon_y)))
print(len(test_pion_y), ",",  len(np.unique(test_pion_y)))
print(len(test_proton_y), ",",  len(np.unique(test_proton_y)))
print("----")
print(
    len(
        np.unique(
            np.concatenate(
                [
                    train_electron_y, test_electron_y,
                    train_muon_y, test_muon_y,
                    train_pion_y, test_pion_y,
                    train_proton_y, test_proton_y
                ]
            )
        )
    )
)
print()
plt.hist(
    train_electron_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[0],
    label="electron",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_electron_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[0],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_pion_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[1],
    label="pion",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_pion_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[1],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_muon_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[2],
    label="muon",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_muon_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[2],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_proton_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[3],
    label="proton",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_proton_y,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[3],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.legend()
plt.ylim(0, 0.001)
plt.savefig("vtx_y.pdf")
plt.close()

train_electron_z = np.load("train_electron_z.npy")
train_muon_z = np.load("train_muon_z.npy")
train_pion_z = np.load("train_pion_z.npy")
train_proton_z = np.load("train_proton_z.npy")
test_electron_z = np.load("test_electron_z.npy")
test_muon_z = np.load("test_muon_z.npy")
test_pion_z = np.load("test_pion_z.npy")
test_proton_z = np.load("test_proton_z.npy")
print(len(train_electron_z), ",",  len(np.unique(train_electron_z)))
print(len(train_muon_z), ",",  len(np.unique(train_muon_z)))
print(len(train_pion_z), ",",  len(np.unique(train_pion_z)))
print(len(train_proton_z), ",",  len(np.unique(train_proton_z)))
print("----")
print(len(test_electron_z), ",",  len(np.unique(test_electron_z)))
print(len(test_muon_z), ",",  len(np.unique(test_muon_z)))
print(len(test_pion_z), ",",  len(np.unique(test_pion_z)))
print(len(test_proton_z), ",",  len(np.unique(test_proton_z)))
print("----")
print(
    len(
        np.unique(
            np.concatenate(
                [
                    train_electron_z, test_electron_z,
                    train_muon_z, test_muon_z,
                    train_pion_z, test_pion_z,
                    train_proton_z, test_proton_z
                ]
            )
        )
    )
)
print()
plt.hist(
    train_electron_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[0],
    label="electron",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_electron_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[0],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_pion_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[1],
    label="pion",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_pion_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[1],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_muon_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[2],
    label="muon",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_muon_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[2],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_proton_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[3],
    label="proton",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_proton_z,
    range=(-1000, 1000), bins=100,
    density=True,
    color=colors[3],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.legend()
plt.ylim(0, 0.001)
plt.savefig("vtx_z.pdf")
plt.close()

# -- Dirs
train_electron_dirx = np.load("train_electron_dirx.npy")
train_muon_dirx = np.load("train_muon_dirx.npy")
train_pion_dirx = np.load("train_pion_dirx.npy")
train_proton_dirx = np.load("train_proton_dirx.npy")
test_electron_dirx = np.load("test_electron_dirx.npy")
test_muon_dirx = np.load("test_muon_dirx.npy")
test_pion_dirx = np.load("test_pion_dirx.npy")
test_proton_dirx = np.load("test_proton_dirx.npy")
print(len(train_electron_dirx), ",",  len(np.unique(train_electron_dirx)))
print(len(train_muon_dirx), ",",  len(np.unique(train_muon_dirx)))
print(len(train_pion_dirx), ",",  len(np.unique(train_pion_dirx)))
print(len(train_proton_dirx), ",",  len(np.unique(train_proton_dirx)))
print("----")
print(len(test_electron_dirx), ",",  len(np.unique(test_electron_dirx)))
print(len(test_muon_dirx), ",",  len(np.unique(test_muon_dirx)))
print(len(test_pion_dirx), ",",  len(np.unique(test_pion_dirx)))
print(len(test_proton_dirx), ",",  len(np.unique(test_proton_dirx)))
print("----")
print(
    len(
        np.unique(
            np.concatenate(
                [
                    train_electron_dirx, test_electron_dirx,
                    train_muon_dirx, test_muon_dirx,
                    train_pion_dirx, test_pion_dirx,
                    train_proton_dirx, test_proton_dirx
                ]
            )
        )
    )
)
print()
plt.hist(
    train_electron_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[0],
    label="electron",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_electron_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[0],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_pion_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[1],
    label="pion",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_pion_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[1],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_muon_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[2],
    label="muon",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_muon_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[2],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_proton_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[3],
    label="proton",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_proton_dirx,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[3],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.legend()
plt.savefig("dir_x.pdf")
plt.close()

train_electron_diry = np.load("train_electron_diry.npy")
train_muon_diry = np.load("train_muon_diry.npy")
train_pion_diry = np.load("train_pion_diry.npy")
train_proton_diry = np.load("train_proton_diry.npy")
test_electron_diry = np.load("test_electron_diry.npy")
test_muon_diry = np.load("test_muon_diry.npy")
test_pion_diry = np.load("test_pion_diry.npy")
test_proton_diry = np.load("test_proton_diry.npy")
print(len(train_electron_diry), ",",  len(np.unique(train_electron_diry)))
print(len(train_muon_diry), ",",  len(np.unique(train_muon_diry)))
print(len(train_pion_diry), ",",  len(np.unique(train_pion_diry)))
print(len(train_proton_diry), ",",  len(np.unique(train_proton_diry)))
print("----")
print(len(test_electron_diry), ",",  len(np.unique(test_electron_diry)))
print(len(test_muon_diry), ",",  len(np.unique(test_muon_diry)))
print(len(test_pion_diry), ",",  len(np.unique(test_pion_diry)))
print(len(test_proton_diry), ",",  len(np.unique(test_proton_diry)))
print("----")
print(
    len(
        np.unique(
            np.concatenate(
                [
                    train_electron_diry, test_electron_diry,
                    train_muon_diry, test_muon_diry,
                    train_pion_diry, test_pion_diry,
                    train_proton_diry, test_proton_diry
                ]
            )
        )
    )
)
print()
plt.hist(
    train_electron_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[0],
    label="electron",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_electron_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[0],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_pion_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[1],
    label="pion",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_pion_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[1],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_muon_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[2],
    label="muon",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_muon_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[2],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_proton_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[3],
    label="proton",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_proton_diry,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[3],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.legend()
plt.savefig("dir_y.pdf")
plt.close()

train_electron_dirz = np.load("train_electron_dirz.npy")
train_muon_dirz = np.load("train_muon_dirz.npy")
train_pion_dirz = np.load("train_pion_dirz.npy")
train_proton_dirz = np.load("train_proton_dirz.npy")
test_electron_dirz = np.load("test_electron_dirz.npy")
test_muon_dirz = np.load("test_muon_dirz.npy")
test_pion_dirz = np.load("test_pion_dirz.npy")
test_proton_dirz = np.load("test_proton_dirz.npy")
print(len(train_electron_dirz), ",",  len(np.unique(train_electron_dirz)))
print(len(train_muon_dirz), ",",  len(np.unique(train_muon_dirz)))
print(len(train_pion_dirz), ",",  len(np.unique(train_pion_dirz)))
print(len(train_proton_dirz), ",",  len(np.unique(train_proton_dirz)))
print("----")
print(len(test_electron_dirz), ",",  len(np.unique(test_electron_dirz)))
print(len(test_muon_dirz), ",",  len(np.unique(test_muon_dirz)))
print(len(test_pion_dirz), ",",  len(np.unique(test_pion_dirz)))
print(len(test_proton_dirz), ",",  len(np.unique(test_proton_dirz)))
print("----")
print(
    len(
        np.unique(
            np.concatenate(
                [
                    train_electron_dirz, test_electron_dirz,
                    train_muon_dirz, test_muon_dirz,
                    train_pion_dirz, test_pion_dirz,
                    train_proton_dirz, test_proton_dirz
                ]
            )
        )
    )
)
print()
plt.hist(
    train_electron_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[0],
    label="electron",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_electron_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[0],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_pion_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[1],
    label="pion",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_pion_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[1],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_muon_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[2],
    label="muon",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_muon_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[2],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_proton_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[3],
    label="proton",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_proton_dirz,
    range=(-1, 1), bins=100,
    density=True,
    color=colors[3],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.legend()
plt.savefig("dir_z.pdf")
plt.close()

# -- Momentum
train_electron_p = np.load("train_electron_p.npy")
train_muon_p = np.load("train_muon_p.npy")
train_pion_p = np.load("train_pion_p.npy")
train_proton_p = np.load("train_proton_p.npy")
test_electron_p = np.load("test_electron_p.npy")
test_muon_p = np.load("test_muon_p.npy")
test_pion_p = np.load("test_pion_p.npy")
test_proton_p = np.load("test_proton_p.npy")
print(len(train_electron_p), ",",  len(np.unique(train_electron_p)))
print(len(train_muon_p), ",",  len(np.unique(train_muon_p)))
print(len(train_pion_p), ",",  len(np.unique(train_pion_p)))
print(len(train_proton_p), ",",  len(np.unique(train_proton_p)))
print("----")
print(len(test_electron_p), ",",  len(np.unique(test_electron_p)))
print(len(test_muon_p), ",",  len(np.unique(test_muon_p)))
print(len(test_pion_p), ",",  len(np.unique(test_pion_p)))
print(len(test_proton_p), ",",  len(np.unique(test_proton_p)))
print("----")
print(
    len(
        np.unique(
            np.concatenate(
                [
                    train_electron_p, test_electron_p,
                    train_muon_p, test_muon_p,
                    train_pion_p, test_pion_p,
                    train_proton_p, test_proton_p
                ]
            )
        )
    )
)
print()
plt.hist(
    train_electron_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[0],
    label="electron",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_electron_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[0],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_pion_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[1],
    label="pion",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_pion_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[1],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_muon_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[2],
    label="muon",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_muon_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[2],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.hist(
    train_proton_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[3],
    label="proton",
    histtype="step", alpha=0.5,
)
plt.hist(
    test_proton_p,
    range=(0, 3000), bins=100,
    density=True,
    color=colors[3],
    histtype="step", alpha=0.5,
    linestyle="dashed"
)
plt.legend()
plt.ylim(0, 0.001)
plt.savefig("momentum.pdf")
plt.close()

# -- Example initial conditions together
train_electron_allinfo = np.column_stack([train_electron_x, train_electron_y, train_electron_z, train_electron_dirx, train_electron_diry, train_electron_dirz, train_electron_p])
print(len(train_electron_allinfo), ",", len(np.unique(train_electron_allinfo, axis=0)))
train_muon_allinfo = np.column_stack([train_muon_x, train_muon_y, train_muon_z, train_muon_dirx, train_muon_diry, train_muon_dirz, train_muon_p])
print(len(train_muon_allinfo), ",", len(np.unique(train_muon_allinfo, axis=0)))
train_pion_allinfo = np.column_stack([train_pion_x, train_pion_y, train_pion_z, train_pion_dirx, train_pion_diry, train_pion_dirz, train_pion_p])
print(len(train_pion_allinfo), ",", len(np.unique(train_pion_allinfo, axis=0)))
train_proton_allinfo = np.column_stack([train_proton_x, train_proton_y, train_proton_z, train_proton_dirx, train_proton_diry, train_proton_dirz, train_proton_p])
print(len(train_proton_allinfo), ",", len(np.unique(train_proton_allinfo, axis=0)))
print("----")
test_electron_allinfo = np.column_stack([test_electron_x, test_electron_y, test_electron_z, test_electron_dirx, test_electron_diry, test_electron_dirz, test_electron_p])
print(len(test_electron_allinfo), ",", len(np.unique(test_electron_allinfo, axis=0)))
test_muon_allinfo = np.column_stack([test_muon_x, test_muon_y, test_muon_z, test_muon_dirx, test_muon_diry, test_muon_dirz, test_muon_p])
print(len(test_muon_allinfo), ",", len(np.unique(test_muon_allinfo, axis=0)))
test_pion_allinfo = np.column_stack([test_pion_x, test_pion_y, test_pion_z, test_pion_dirx, test_pion_diry, test_pion_dirz, test_pion_p])
print(len(test_pion_allinfo), ",", len(np.unique(test_pion_allinfo, axis=0)))
test_proton_allinfo = np.column_stack([test_proton_x, test_proton_y, test_proton_z, test_proton_dirx, test_proton_diry, test_proton_dirz, test_proton_p])
print(len(test_proton_allinfo), ",", len(np.unique(test_proton_allinfo, axis=0)))
print()

print(len(np.unique(np.concatenate([np.unique(train_electron_allinfo, axis=0), np.unique(test_electron_allinfo, axis=0)]), axis=0)))
print(len(np.unique(np.concatenate([np.unique(train_muon_allinfo, axis=0), np.unique(test_muon_allinfo, axis=0)]), axis=0)))
print(len(np.unique(np.concatenate([np.unique(train_pion_allinfo, axis=0), np.unique(test_pion_allinfo, axis=0)]), axis=0)))
print(len(np.unique(np.concatenate([np.unique(train_proton_allinfo, axis=0), np.unique(test_proton_allinfo, axis=0)]), axis=0)))
print()

print(len(np.unique(np.concatenate([np.unique(train_electron_allinfo, axis=0), np.unique(train_pion_allinfo, axis=0), np.unique(train_muon_allinfo, axis=0)]), axis=0)))
