from Main_with_rotation import Config, main


def run_demo():
    """Generate example slices at reduced resolution and save to examples/ directory."""
    cfg = Config(res_hi=256, num_rotated=4, output_dir="examples")
    main(cfg)


if __name__ == "__main__":
    run_demo()
