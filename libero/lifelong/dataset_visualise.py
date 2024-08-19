import h5py
import cv2
import numpy as np
import argparse

def main():
    # Open the H5 file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:

        agentview_images = f["data/demo_0/obs/agentview_rgb"]
     
        # Display each image in the sequence
        for img in agentview_images:
            # Make video upside down.
            # img = np.flipud(img)
            cv2.imshow("AgentView Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()