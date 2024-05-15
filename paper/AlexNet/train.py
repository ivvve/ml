from model import AlexNet
from data import load_train_data
import torch
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)

    model = AlexNet().to(device)
    model = torch.nn.parallel.DataParallel(model, device_ids=[0, 1, 2, 3])
    summary(model, (3, 227, 227))

    seed = torch.initial_seed()
    print(f"{seed=}")

    tbwritter = SummaryWriter(log_dir="./tb_logs")

    MOMENTUM = 0.9
    LR_DECAY = 0.0005
    LR_INIT = 0.01
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=LR_DECAY,
    )
    # decrease LR by 1/10 every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_dataloader = load_train_data()
    NUM_EPOCHS = 90
    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        for X, y in train_dataloader:
            total_steps += 1

            X = X.to(device)
            y = y.to(device)
            
            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tensorboard log
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(y_pred, 1)
                    accuracy = torch.sum(preds == y)
                    
                    tbwritter.add_scalar("loss", loss.item(), total_steps)
                    tbwritter.add_scalar("accuracy", accuracy.item(), total_steps)
        
        lr_scheduler.step()

        # save checkpoint
        checkpoint_path = os.path.join(".", f"alexnet_states_e{epoch+1}.pkl")
        torch.save({
            "epoch": epoch,
            "total_steps": total_steps,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "seed": seed,
        }, checkpoint_path)
