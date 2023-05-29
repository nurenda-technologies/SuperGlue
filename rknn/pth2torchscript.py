import torch
import argparse
from superglue.superpoint import *


def main(args):
    state_dict = torch.load(args.weights)

    for key in state_dict.keys():
        print("{}: {}".format(key, state_dict[key].shape))

    encoder = SuperPointEncoder()
    encoder_state_dict = {
        "conv1a.weight": state_dict["conv1a.weight"],
        "conv1a.bias": state_dict["conv1a.bias"],
        "conv1b.weight": state_dict["conv1b.weight"],
        "conv1b.bias": state_dict["conv1b.bias"],
        "conv2a.weight": state_dict["conv2a.weight"],
        "conv2a.bias": state_dict["conv2a.bias"],
        "conv2b.weight": state_dict["conv2b.weight"],
        "conv2b.bias": state_dict["conv2b.bias"],
        "conv3a.weight": state_dict["conv3a.weight"],
        "conv3a.bias": state_dict["conv3a.bias"],
        "conv3b.weight": state_dict["conv3b.weight"],
        "conv3b.bias": state_dict["conv3b.bias"],
        "conv4a.weight": state_dict["conv4a.weight"],
        "conv4a.bias": state_dict["conv4a.bias"],
        "conv4b.weight": state_dict["conv4b.weight"],
        "conv4b.bias": state_dict["conv4b.bias"],
    }
    encoder.load_state_dict(encoder_state_dict)
    encoder_script = torch.jit.script(encoder)
    torch.jit.save(encoder_script, "weights/superpoint_encoder.pt")

    detector = SuperPointDetector()
    detector_state_dict = {
        "convPa.weight": state_dict["convPa.weight"],
        "convPa.bias": state_dict["convPa.bias"],
        "convPb.weight": state_dict["convPb.weight"],
        "convPb.bias": state_dict["convPb.bias"],
    }
    detector.load_state_dict(detector_state_dict)
    detector_script = torch.jit.script(detector)
    torch.jit.save(detector_script, "weights/superpoint_detector.pt")

    describer = SuperPointDescriber()
    describer_state_dict = {
        "convDa.weight": state_dict["convDa.weight"],
        "convDa.bias": state_dict["convDa.bias"],
        "convDb.weight": state_dict["convDb.weight"],
        "convDb.bias": state_dict["convDb.bias"],
    }
    describer.load_state_dict(describer_state_dict)
    describer_script = torch.jit.script(describer)
    torch.jit.save(describer_script, "weights/superpoint_describer.pt")

    superpoint = SuperPoint()
    superpoint.encoder.load_state_dict(encoder_state_dict)
    superpoint.detector.load_state_dict(detector_state_dict)
    superpoint.describer.load_state_dict(describer_state_dict)
    superpoint_script = torch.jit.script(superpoint)
    torch.jit.save(superpoint_script, "weights/superpoint.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="Path to trained PyTorch .pth model weights to convert.",
    )
    args = parser.parse_args()

    main(args)
