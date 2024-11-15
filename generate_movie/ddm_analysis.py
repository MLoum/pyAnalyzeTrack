import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def analyze_ddm(images: np.ndarray, buffer_size: int, camera: object, dt: float, D_guess: float, lightsheet: bool, radius: float, fps: int,
                model_id: int = 6, print_report: bool = True, show_graphs: bool = True, show_interactive_radavg: bool = True, save_report: bool = True) -> None:
    """
    Function that analyze a stack of images using DDM analysis from ddm-toolkit (https://github.com/mhvwerts/ddm-toolkit/tree/master)

    :param images: images of the video
    :param buffer_size: buffer size for the DDM must be less than the number of images
    :param camera: Camera used for the video
    :param dt: timestep of the video
    :param D_guess: initial guess of the diffusion coefficient (will be refined)
    :param lightsheet: if we save the report it will be written in it else it useless
    :param radius: if we save the report it will be written in it else it's useless
    :param fps: if we save the report it will be written in it else it's useless
    :param model_id: model id of the model
    :param print_report: print the report of the DDM analysis
    :param show_graphs: show graphs of the DDM analysis
    :param save_report: save the report of the DDM analysis (graphs and their data are not saved)
    """

    # Try to import ddm-toolkit
    try:
        sys.path.insert(0, '../ddm-toolkit-master')
        from ddm_toolkit import ddm
        from ddm_toolkit.analysis import ISFanalysis_simple_brownian
    except ModuleNotFoundError:
        sys.exit('ddm_toolkit_master not found.')

    # Select the engine
    engine = ddm.ImageStructureEngine(camera.size, buffer_size, model_id)

    # Push images in the engine to analyze them
    for i, image in enumerate(images):
        engine.push(image)

    # Create ImageStructureFunction from the image
    ISF = ddm.ImageStructureFunction.fromImageStructureEngine(engine)

    # Set the real parameters used for the video (size of pixel and timestep)
    ISF.real_world(camera.pixel_to_size, dt)

    # Get result from an initial guess
    result = ISFanalysis_simple_brownian(ISF, D_guess)

    if print_report:
        result.print_report()

    # Save report in the file "ddm_results.txt"
    if save_report:
        with open("ddm_results.txt", "a") as file:
            file.write(
                f"{radius}\t{lightsheet}\t{fps}\t{result.D_guess}\t{result.D_guess_refined}\t{result.q_low}\t{result.q_opt}\t{result.q_high}\t{result.q[-1]}\t{result.D_fit}\t{result.D_fit_CI95}\n")

    if show_graphs:
        # Show graphs returned by ddm-toolkit
        result.show_ISF_radavg()
        result.show_fits()
        result.show_Aq_Bq_kq()

    # Show an interactive graph the see the radial average for different time difference between images
    if show_interactive_radavg:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        fig.canvas.manager.set_window_title("Interactive ISF and radial average")

        r = (ISF.bins[:-1] + ISF.bins[1:]) / 2
        ys = []
        for y in range(buffer_size + 1):
            ys.append(ISF.radavg(y))
        ax_r = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        ax1.set_xlabel('q [µm-1]')
        ax1.set_ylabel('ISF')

        ax2.set_title('ISF')

        N_slider = Slider(
            ax=ax_r,
            label='timestep difference',
            valmin=0,
            valmax=buffer_size,
            valinit=buffer_size,
            valstep=1
        )

        ISF_images = engine.ISF()
        line, = ax1.plot(r, ys[buffer_size])
        ax2.imshow(ISF_images[buffer_size])
        fig.subplots_adjust(bottom=0.25)

        def update(val):
            line.set_ydata(ys[int(val)])
            ax2.cla()
            ax2.imshow(ISF_images[val])
            fig.canvas.draw_idle()

        N_slider.on_changed(update)

    if show_interactive_radavg or show_graphs:
        plt.show()
