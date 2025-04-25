from ultralytics import RTDETR

def fine_tuning():
    # Load pre-trained RT-DETR model
    model = RTDETR("weights/rtdetr-x.pt").to("cuda")

    # Set training parameters
    model.train(data='SKU-110K.yaml', epochs=50, batch=8, imgsz=640, device='cuda', workers=2, patience=10)

    # Evaluate on validation set
    metrics = model.val()
    print("metrics: ", metrics)

    # Save the fine-tuned model
    model.save('weights/rtdetr-x-sku110k-2.pt')


if __name__ == '__main__':
    fine_tuning()
