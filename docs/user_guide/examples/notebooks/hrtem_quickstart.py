import abtem
import ase
import matplotlib.pyplot as plt
import numpy as np

abtem.config.set(
    {
        "device": "gpu",
        "fft": "fftw",
        "diagnostics.task_progress": True,
        "diagnostics.progress_bar": "tqdm",
    }
)
dir = "/media/chenguisen/WD_BLACK/cgs/group/chenjing/Phase-simulation"
import os

files = os.listdir(dir)


for i in files:
    if i.endswith(".cif"):
        #print(i)
        file = os.path.join(dir, i)
        print(file)
        tubes = ase.io.read(file)
        tubes = tubes * (1, 1, 2)
        rotated_tubes = tubes.copy()

        # rotate cell and atoms by 90 degrees around y
        #rotated_tubes.rotate("y", 90, rotate_cell=True)

        # standardize unit cell (done automatically in an abTEM simulation)
        rotated_tubes = abtem.standardize_cell(rotated_tubes)

        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        #abtem.show_atoms(rotated_tubes, plane="xy", ax=ax1, title="Beam view")
        #abtem.show_atoms(rotated_tubes, plane="yz", ax=ax2, title="Side view")
        frozen_phonons = abtem.FrozenPhonons(rotated_tubes, 32, sigmas=0.1)
        potential = abtem.Potential(
                                    frozen_phonons,
                                    sampling=0.05,
                                    projection="infinite",
                                    slice_thickness=1,
                                )
        wave = abtem.PlaneWave(energy=300e3)
        exit_wave = wave.multislice(potential)
        exit_wave.compute()
        Cs = -8e-6 * 1e10  # spherical aberration (-8 um)

        ctf = abtem.CTF(Cs=Cs, energy=wave.energy, defocus="scherzer")

        print(f"defocus = {ctf.defocus:.2f} Å")
        ctf.semiangle_cutoff = ctf.crossover_angle
        Cc = 1.0e-3 * 1e10  # chromatic aberration (1.2 mm)
        energy_spread = 0.35  # standard deviation energy spread (0.35 eV)

        focal_spread = Cc * energy_spread / exit_wave.energy

        incoherent_ctf = ctf.copy()
        incoherent_ctf.focal_spread = focal_spread
        measurement_ensemble = exit_wave.apply_ctf(incoherent_ctf).intensity()

        measurement_ensemble.shape
        measurement_ensemble._area_per_pixel

        measurement = measurement_ensemble.mean(0)

        #measurement.show(cmap="gray", title="Simulated HRTEM image")

        filtered_measurements = measurement.gaussian_filter(sigma=0.7)
        #filtered_measurements.show(cmap="gray", title="Blurred HRTEM image")

        fig, ax = plt.subplots(figsize=(8, 8))
        filtered_measurements.show(ax=ax, cmap="gray")
        print(filtered_measurements.sampling)
        with open(os.path.join(dir, "measurements_log.txt"), "a") as f:
            f.write(f"{i}: {filtered_measurements.sampling}\n")
        ax.axis('off')  # 隐藏坐标轴和刻度
        plt.tight_layout()  # 自动调整布局，防止图形超出边界
        plt.savefig(os.path.join(dir, i.split(".")[0] + ".png"), bbox_inches='tight', pad_inches=0, dpi=300)
        #plt.close()  # 关闭图形释放内存
