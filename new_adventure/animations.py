import matplotlib.pyplot as plt
import seaborn as sns

import shutil, os
from utils import *
from save_load import *



def create_animation_1d_pictures_particles(x_paths, potential_paths, X, Y, ani_path, graph_details={"p_size": 1}):
    num_steps, num_particles, dim = x_paths.shape

    for t in range(num_steps):

        # so we can reuse the axis
        f, (ax_hist, ax_graph) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.2, .8)}, figsize=(14,12))

        sns.distplot(x_paths[t], ax=ax_hist)
        sns.scatterplot(x_paths[t].reshape(-1), potential_paths[t], ax=ax_graph)
        sns.lineplot(x=X, y=Y, ax=ax_graph)

        plt.savefig(os.path.join(ani_path , "{}.png".format(t)))

        plt.close()

def create_animation_2d_pictures_particles(all_paths, X, Y, Z, ani_path, graph_details={"p_size": 1, "density_function": None}):
    """path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2] = path_z"""

    available_colors = ["red", "green"]

    num_tau, num_particles, tau, dim = all_paths.shape

    density_function = graph_details["density_function"]

    for i in range(len(all_paths)):
        curr_paths = all_paths[i]


        if density_function is not None:
            fig = plt.figure(figsize=(30, 10))
            gs = fig.add_gridspec(1, 4)
            ax = fig.add_subplot(gs[:, 0:2])
            ax2 = fig.add_subplot(gs[:, 2:4])
        else:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(1, 1, 1)

        if density_function is not None:
            inp = np.array(np.meshgrid(X, Y)).reshape(2, len(X) * len(Y))
            Z_density = density_function(inp, curr_paths[:, j, :2]).reshape(len(X), len(Y))

            if graph_details["type"] == "contour":
                ax2.contour(X, Y, Z_density, 40)
            else:
                ax2.imshow(Z_density, cmap=plt.cm.gist_earth_r, extent=[X[0], X[-1], Y[-1], Y[0]],
                            interpolation=graph_details["interpolation"])
        else:
            ax.plot(curr_paths[:, j, 0], curr_paths[:, j, 1], "o", color=color_use, markersize=graph_details["p_size"])

        # fig.suptitle(folder_name, fontsize=20)

        if graph_details["type"] == "contour":
            ax.contour(X, Y, Z, 40)
        else:
            ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0], X[-1], Y[-1], Y[0]],
                        interpolation=graph_details["interpolation"])


        plt.savefig(os.path.join(ani_path , "{}.png".format(i * tau + j)))

        plt.close()

def get_potential_path(all_paths, F):
    flat_paths = all_paths.reshape(-1, all_paths.shape[2])
    Y = F.f(flat_paths)
    return Y.reshape(all_paths.shape[0], all_paths.shape[1])

def get_animation(x_paths, config, animations_path):

    # get supplementary stuff
    d = config["domain_dim"]

    F = get_potential(config)
    potential_paths = get_potential_path(x_paths, F)
    animation_cache_path = os.path.join(animations_path, "animation_cache")
    if not os.path.isdir(animation_cache_path):
        os.makedirs(animation_cache_path)

    if d == 1:
        # get the function plot inputs
        X = np.linspace(config["x_range"][0], config["x_range"][1], 200)
        inp = np.array([X]).T
        Y = F.f(inp)

        create_animation_1d_pictures_particles(x_paths, potential_paths, X, Y, animation_cache_path,
                                                            graph_details={"p_size": 3})


    elif d == 2:
        # get the function plot inputs

        X = np.linspace(process["x_range"][0], process["x_range"][1], 100)
        Y = np.linspace(process["x_range"][0], process["x_range"][1], 100)
        inp = np.array(np.meshgrid(X, Y)).reshape(2, len(X) * len(Y))

        Z = f(inp).reshape(len(X), len(Y))


        # full process densities
        K = multi_gaussian(np.array([[0.6, 0], [0, 0.6]]))

        create_animation_2d_pictures_particles(all_paths_reordered, X, Y, Z, ani_path,
                                                            graph_details={"type": "heatmap", "p_size": 3,
                                                                           # "density_function": None})
                                                                           "density_function": lambda inp, p: V(inp, K, p),
                                                                       "interpolation": "bilinear"})
    else:
        print(d)
        raise Exception("Error: Dimension {} does not exist.".format(d))


    create_animation(animation_cache_path, os.path.join(animations_path, "{}.mp4".format("animation")), framerate=2)

    time.sleep(3)  # otherwise it deletes the images before getting the video
    shutil.rmtree(animation_cache_path)  # remove dir and all contains


def get_exp_animations(exp_folder):

    for process_id, animation_path in animation_path_generator(exp_folder):

        print(process_id)

        all_paths = load_opt_path(exp_folder, process_id)
        config = load_config(exp_folder, process_id)
        
        get_animation(all_paths, config, animation_path)


# ffmpeg -r 20 -f image2 -s 1920x1080 -i %d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
def create_animation(image_folder, video_path, screen_resolution="1920x1080", framerate=30, qaulity=25,
                     extension=".png"):
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-r", str(framerate),
            "-f", "image2",
            "-s", screen_resolution,
            "-i", os.path.join(image_folder, "%d" + extension),
            "-vcodec", "libx264",
            "-crf", str(qaulity),
            "-pix_fmt", "yuv420p",
            video_path
        ])

if __name__ == "__main__":

    root_folder = os.environ["PATH_TO_ADV_FOLDER"]
    data_name = "quadratic"
    exp_name = "ISMD_Oct28_17-55-02_Daniels-MacBook-Pro-4.local"
    experiment_folder = os.path.join(root_folder, "experiments", data_name, exp_name)

    get_exp_animations(experiment_folder)

    