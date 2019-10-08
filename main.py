import fire
import models 
import os
import torch as t
from config import opt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
#opt = DefaultConfig()
#lr = opt.lr
#model = getattr(models, opt.model)
#dataset = DogCat(opt.train_data_root)

def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)
    
    model = getattr(models,opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: 
        model.cuda()
    # 数据设定  户籍科 010 82640433
    train_data = DogCat(opt.load_model_path,train=True)
    val_data = DogCat(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
        shuffle = True,
        num_workers = opt.num_workers
        )
    train_dataloader = DataLoader(test_data,opt.batch_size,
        shuffle = False,
        num_workers = opt.num_workers
        )
    # 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model)
    # 统计指标，平滑处理之后的损失
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in tqdm(enumerate(train_dataloader)): # ii num ,(data,label) enumerate
            # 训练模型参数
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.stop()
            # 更新统计指标及可视化
            loss_meter.add(loss.data[0])
            confusion_matrix.add(loss.data[0])
            confusion_matrix.add(score.data,target.data)
            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss',loss_meter.value()[0])
                
                if os.path.exist(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()
            model.save()

            # 计算验证集上的指标及其可视化
            val_cm,val_accuracy = val(model,val_dataloader)
            vis.plot('val_accuracy',val_accuracy)
            vis.log('epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}'
                .format(
                    epoch = epoch,
                    loss = loss_meter.value()[0],
                    val_cm = str(val_cm.value()),
                    train_cm = str(confusion_matrix.value()),
                    lr = lr
                ))
            if loss_meter.value()[0] > previous_loss:
                lr = lr * opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                previous_loss = loss_meter.value()[0]
def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input , label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.long(),volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(),label.long())

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1] ) / cm_value.sum()
    return confusion_matrix, accuracy
def test(**kwargs):
    opt.parse(kwargs)
    model = getattr(models,opt.model)().model.eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,
        batch_size = opt.batch_size,
        shuffle = False, 
        num_workers = opt.num_workers
        )
    results = []
    for ii ,(data,path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data,volatile = True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:,1].data.tolist()
        batch_results = [(path_,probaility_)
            for path_,probability_ in zip(path,probaility)]
        results += batch_results
    write_csv(results,opt.result_file)
    return results

def help():
    print('help')
if __name__=='__main__':
    import fire
    fire.Fire()