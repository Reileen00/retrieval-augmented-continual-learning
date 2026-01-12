import torch
import torch.nn as nn
from model import ContinualTransformer
from datasets import get_tasks
from memory import Memory
from retrieve import embed

model = ContinualTransformer()
opt = torch.optim.Adam(model.parameters(),1e-4)
mem = Memory()

tasks = get_tasks()

for t,task in enumerate(tasks):
    print("Training task",t)
    for x,y in task:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        emb = embed(model,x)

        # retrieve relevant memory
        mx,my = mem.sample(emb)

        if len(mx)>0:
            rx = torch.stack(mx)
            ry = torch.stack(my)
            x = torch.cat([x,rx])
            y = torch.cat([y,ry])

        pred = model(x)
        loss = nn.CrossEntropyLoss()(pred.view(-1,1000),y.view(-1))

        loss.backward()
        opt.step()
        opt.zero_grad()

        mem.add(emb,x[0],y[0])
