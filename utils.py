import matplotlib.pyplot as plt


def summary(results, output_prefix="results"):
    # Sort by num_blocks for clean reporting
    results = sorted(results, key=lambda d: d["num_blocks"])
    
    print("\n=== Summary ===")
    for r in results:
        print(f"num_blocks={r['num_blocks']:2d}  elapsed={r['elapsed']:.4f} sec  acc={r['acc']:.4f}  Used_GPU_num={r['Used_GPU_num']}")
    # print total elapsed time for all runs
    times = [r["elapsed"] for r in results]
    total_time = sum(times)
    print(f"Total elapsed time for all runs: {total_time:.4f} sec")
    
    accuracies = [r["acc"] for r in results]
    used_gpus = [r["Used_GPU_num"] for r in results]
    
    f, a = plt.subplots(1, 3, figsize=(17, 4))
    a[0].plot([r["num_blocks"] for r in results], times)
    a[0].set_xlabel("num_blocks")
    a[0].set_ylabel("elapsed time (sec)")
    a[0].set_title("Training Time vs. num_blocks")  
    a[1].plot([r["num_blocks"] for r in results], accuracies)
    a[1].set_xlabel("num_blocks")
    a[1].set_ylabel("accuracy")
    a[1].set_title("Validation Accuracy vs. num_blocks")
    a[2].plot([r["num_blocks"] for r in results], used_gpus, marker='o')
    a[2].set_xlabel("num_blocks")
    a[2].set_ylabel("Used_GPU_num")
    a[2].set_title("Used GPU Number vs. num_blocks")
    
    a[0].set_xticks([r["num_blocks"] for r in results])
    a[1].set_xticks([r["num_blocks"] for r in results])
    a[2].set_xticks([r["num_blocks"] for r in results])
    

    a[1].set_ylim(0.9,1.0)
    plt.tight_layout()

    f.savefig(f"Figures/{output_prefix}.png")
    for idx, ax in enumerate(a, start=1):
        single_fig, single_ax = plt.subplots(figsize=(5, 4))
        for line in ax.get_lines():
            single_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                marker=line.get_marker(),
                linestyle=line.get_linestyle(),
                color=line.get_color(),
            )
        single_ax.set_xlabel(ax.get_xlabel())
        single_ax.set_ylabel(ax.get_ylabel())
        single_ax.set_title(ax.get_title())
        single_ax.set_xticks(ax.get_xticks())
        single_ax.set_ylim(ax.get_ylim())
        single_fig.tight_layout()
        single_fig.savefig(f"Figures/{output_prefix}_plot_{idx}.png")
        plt.close(single_fig)

    plt.show()
    