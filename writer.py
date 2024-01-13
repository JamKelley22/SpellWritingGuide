"""Module to manage encoding, decoding, and writing of spells"""
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import bases
import line_shapes

cmap = plt.get_cmap("Dark2")


def load_attribute(fname):
    with open(fname, "r", encoding="utf-8") as f:
        data = f.readlines()
        f.close()
    data = [d.replace("\n", "").lower() for d in data]
    return data


levels = sorted(load_attribute("Attributes/levels.txt"), key=int)
ranges = sorted(load_attribute("Attributes/range.txt"))
area_types = sorted(load_attribute("Attributes/area_types.txt"))
damage_types = sorted(load_attribute("Attributes/damage_types.txt"))
schools = sorted(load_attribute("Attributes/school.txt"))


# ---------Functions for creating unique binary numbers------
def cycle_list(l, loops=1):
    n = len(l)
    for t in range(loops):
        l = [l[(i + 1) % n] for i in range(n)]
    return l


def generate_unique_combinations(L):
    combinations = generate_binary_strings(L)
    non_repeating = [combinations[0]]
    for i in tqdm(range(len(combinations)), desc="Genearting Unique Binary Numbers"):
        ref = list(combinations[i])
        N = len(ref)
        test = 0
        for j in range(len(non_repeating)):
            for n in range(N):
                if cycle_list(list(non_repeating[j]), loops=n + 1) == ref:
                    test += 1

        if test == 0:
            non_repeating.append(combinations[i])

    for i in np.arange(len(non_repeating)):
        non_repeating[i] = [int(s) for s in list(non_repeating[i])]
    return non_repeating


def genbin(n, bs=""):
    if n - 1:
        genbin(n - 1, bs + "0")
        genbin(n - 1, bs + "1")
    else:
        print("1" + bs)


def generate_binary_strings(bit_count):
    binary_strings = []

    def genbin(n, bs=""):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + "0")
            genbin(n, bs + "1")

    genbin(bit_count)
    return binary_strings


# -------Functions for drawing runes
def decode_shape(
    in_array,
    k=1,
    point_color="k",
    color="k",
    label=None,
    base_fn=bases.polygon,
    base_kwargs=None,
    shape_fn=line_shapes.straight,
    shape_kwargs=None,
    plot_base=False,
    hide_dotted=False,
):
    # decodes a single array into a given base, use plot_base = True if you are plotting it on its own
    if base_kwargs is None:
        base_kwargs = []
    if shape_kwargs is None:
        shape_kwargs = []

    n = len(in_array)
    x, y = base_fn(n, *base_kwargs)
    if plot_base:
        plt.scatter(x[1:], y[1:], s=70, facecolors="none", edgecolors=point_color)
        plt.scatter(x[0], y[0], s=70, facecolors=point_color, edgecolors=point_color)
        plt.axis("off")
        plt.axis("scaled")
    for i, elem in enumerate(in_array):
        P = [x[i], y[i]]
        Q = [x[(i + k) % n], y[(i + k) % n]]
        X, Y = shape_fn(P, Q, *shape_kwargs)
        # Controls showing connection points as dotted
        if not hide_dotted:
            if elem == 0:
                plt.plot(X, Y, color=color, ls=":", linewidth=0.5)
        if elem == 1:
            plt.plot(
                X,
                Y,
                color=color,
                ls="-",
                label=label if i == np.where(in_array == 1)[0][0] else None,
            )
        # else:
        #     print(f"elem {elem} at index {i} is not valid, input being skipped")


def draw_multiple_inputs(
    in_array,
    base_fn=bases.polygon,
    base_kwargs=None,
    shape_fn=line_shapes.straight,
    shape_kwargs=None,
    point_color="k",
    labels=None,
    legend=False,
    colors=None,
    legend_loc="upper left",
    hide_points=False,
    hide_dotted=False,
):
    # draws multiple inputs on a single base
    if base_kwargs is None:
        base_kwargs = []
    if shape_kwargs is None:
        shape_kwargs = []
    if labels is None:
        labels = []
    if colors is None:
        colors = []

    if colors == []:
        colors = [point_color] * in_array.shape[0]
    n = in_array.shape[1]
    x, y = base_fn(n, *base_kwargs)
    if not hide_points:
        # Starting Point
        plt.scatter(x[1:], y[1:], s=70, facecolors="none", edgecolors=point_color)
        # Remainder of points
        plt.scatter(x[0], y[0], s=70, facecolors=point_color, edgecolors=point_color)

    if len(labels) != in_array.shape[0]:
        labels = [None] * in_array.shape[0]

    for i, k in enumerate(range(in_array.shape[0])):
        decode_shape(
            in_array[i],
            k=k + 1,
            base_fn=base_fn,
            base_kwargs=base_kwargs,
            shape_fn=shape_fn,
            shape_kwargs=shape_kwargs,
            label=labels[i],
            color=colors[i],
            hide_dotted=hide_dotted,
        )
    if labels[0] != None and legend == True:
        plt.legend(loc=legend_loc, fontsize=4)
    plt.axis("off")
    plt.axis("scaled")


def draw_spell(
    level,
    rang,
    area,
    damage_type,
    school,
    title=None,
    output="output.png",
    legend=False,
    base_fn=bases.polygon,
    base_kwargs=None,
    shape_fn=line_shapes.straight,
    shape_kwargs=None,
    colors=None,
    legend_loc="upper left",
    breakdown=False,
    hide_points=False,
    hide_dotted=False,
):
    # draws a spell given certain values by comparing it to input txt
    if base_kwargs is None:
        base_kwargs = []
    if shape_kwargs is None:
        shape_kwargs = []
    if colors is None:
        colors = []

    i_range = ranges.index(rang)
    i_levels = levels.index(level)
    i_area = area_types.index(area)
    i_damage_type = damage_types.index(damage_type)
    i_school = schools.index(school)
    attributes = [i_levels, i_school, i_damage_type, i_area, i_range]
    labels = [
        f"level: {level}",
        f"school: {school}",
        f"damage type: {damage_type}",
        f"range: {rang}",
        f"area_type: {area}",
    ]
    N = 2 * len(attributes) + 1

    if len(colors) == 0 and breakdown == True:
        colors = [cmap(i / len(attributes)) for i in range(len(attributes))]
    if not os.path.isdir("Uniques/"):
        os.makedirs("Uniques/")
    if os.path.isfile(f"Uniques/{N}.npy"):
        non_repeating = np.load(f"Uniques/{N}.npy")
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy", non_repeating)
    input_array = np.array(
        [non_repeating[i] for i in attributes]
    )  # note +1 s.t. 0th option is always open for empty input
    draw_multiple_inputs(
        input_array,
        labels=labels,
        legend=legend,
        base_fn=base_fn,
        base_kwargs=base_kwargs,
        shape_fn=shape_fn,
        shape_kwargs=shape_kwargs,
        colors=colors,
        legend_loc=legend_loc,
        hide_points=hide_points,
        hide_dotted=hide_dotted,
    )
    plt.title(title)
    plt.savefig(output, dpi=250)


if __name__ == "__main__":
    base_fn_mapping = {
        "circle": bases.circle,
        "polygon": bases.polygon,
        "cubic": bases.cubic,
        "golden": bases.golden,
        "line": bases.line,
        "quadratic": bases.quadratic,
    }
    line_shape_fn_mapping = {
        "centre_circle": line_shapes.centre_circle,
        "non_centre_circle": line_shapes.non_centre_circle,
        "straight": line_shapes.straight,
    }

    parser = argparse.ArgumentParser(
        epilog="Additional text to include in --help message.",
    )

    parser.add_argument(
        "--level",
        "-l",
        help="level of the spell. Allowed values are " + ", ".join(levels),
        default="3",
        choices=levels,
        metavar="<level option>",
    )
    parser.add_argument(
        "--range",
        "-r",
        help="range of the spell. Allowed values are " + ", ".join(ranges),
        default="point (150 feet)",
        choices=ranges,
        metavar="<range option>",
    )
    parser.add_argument(
        "--area",
        "-a",
        help="area type of the spell. Allowed values are " + ", ".join(area_types),
        default="sphere",
        choices=area_types,
        metavar="<area option>",
    )
    parser.add_argument(
        "--damage_type",
        "-d",
        help="Damage type of the spell. Allowed values are " + ", ".join(damage_types),
        default="fire",
        choices=damage_types,
        metavar="<damage type option>",
    )
    parser.add_argument(
        "--school",
        "-s",
        help="school of the spell. Allowed values are " + ", ".join(schools),
        default="evocation",
        choices=schools,
        metavar="<school option>",
    )
    parser.add_argument("--title", "-t", help="title in plot")
    parser.add_argument("--output", "-o", help="output file name", default="output.png")
    parser.add_argument("--legend", help="print legend", action="store_true")
    parser.add_argument(
        "--breakdown", help="breakdown lines with colour", action="store_true"
    )
    parser.add_argument(
        "--base", choices={*list(base_fn_mapping.keys())}, default="polygon"
    )
    parser.add_argument(
        "--line_shape",
        choices={*list(line_shape_fn_mapping.keys())},
        default="straight",
    )
    parser.add_argument("-b", default={1})
    parser.add_argument(
        "--hide_points", help="hide points that make up base shape", action="store_true"
    )
    parser.add_argument(
        "--hide_dotted", help="hide dotted connectors", action="store_true"
    )

    args = parser.parse_args()

    draw_spell(
        level=args.level,
        rang=args.range,
        area=args.area,
        damage_type=args.damage_type,
        school=args.school,
        title=args.title,
        legend=args.legend,
        base_fn=base_fn_mapping[args.base],
        shape_fn=line_shape_fn_mapping[args.line_shape],
        breakdown=args.breakdown,
        output=args.output,
        hide_points=args.hide_points,
        hide_dotted=args.hide_dotted,
    )
    # plt.clf()
    # input_shape = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    # decode_shape(input_shape,k=3,point_color = 'k',color = 'k',
    #          label = None,base_fn = bases.polygon,base_kwargs = [],
    #          shape_fn = line_shapes.straight,shape_kwargs = [],
    #          plot_base = True)
    # plt.axis('off')
    # plt.savefig("test.png")
