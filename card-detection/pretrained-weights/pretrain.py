from ultralytics import YOLO
import torch

def main():
    model = YOLO("Z:\coding\PR project\card-detection\pretrained-weights\yolov8m_tuned.pt")


    # print(model.model.model)
    detect_layer_index = 22

    for name, param in model.model.named_parameters():
        layer_idx = int(name.split(".")[1]) 
        if layer_idx == detect_layer_index:
            param.requires_grad = True # freeze 22nd layer
        else:
            param.requires_grad = False # freeze others

    print(torch.cuda.is_available())

    # Fine-tune on your 32-class dataset
    model.train(
        data="Z:\coding\PR project\dataset\dataset.yaml",   # defines nc: 32 and your class names
        patience=20,  # instead of guessing epochs, it stops training when the loss converges
        epochs=30,
        imgsz=640,
        pretrained=True,
        # device=0
    )

if __name__ == "__main__":
    main()