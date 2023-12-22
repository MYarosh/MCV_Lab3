import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
import time, tracemalloc, argparse, warnings

import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F

CLASSES = 6 # Число выводимых вероятных классов 

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('file', nargs='?')
    return parser

warnings.filterwarnings("ignore")

# Чтение названия файла изображения из аргументов
parser = createParser()
namespace = parser.parse_args()
if not namespace.file:
    print("No file argument")
    exit()

#Загрузка названий классов изображений
with open('imagenet_classes.txt') as labels:
    classes = [i.strip() for i in labels.readlines()]

#Преобразования изображения
transforms = transforms.Compose([transforms.Resize((256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


#Чтение изображение и его преобразование в тензор
image_predict = Image.open(namespace.file)
tensor_image = transforms(image_predict)
x1 = torch.unsqueeze(tensor_image, 0).cuda()


#Regular model
print("Regular model\n")
tracemalloc.start()

times = 0
for i in range(5):
    #Загрузка модели
    timest = time.time()
    model = alexnet(pretrained=True).eval().cuda()
    cur_times = time.time()-timest
    print("Load time regular model {}".format(cur_times))
    times += cur_times
print("Mean load time {}\n".format(times/5))

times = 0
for i in range(5):
    #Запуск модели
    timest = time.time()
    output = model(x1)
    cur_times = time.time()-timest
    print("Time to predict: {}".format(cur_times))
    times += cur_times
print("Mean predict time {}\n".format(times/5))

#Преобразование результатов в проценты и сортировка их
_, indices = torch.sort(output, descending = True)
percentage = F.softmax(output, dim=1)[0] * 100.0

#Вывод результатов в консоль
print("{} most probable classes for {}:".format(CLASSES, namespace.file))
for idx in indices[0][:5]:
    print('{}: {:.3f}%'.format(classes[idx], percentage[idx].item()))

#Вывод максимума используемой памяти 
print('Peak of used memory {}\n'.format(tracemalloc.get_traced_memory()[1]))
tracemalloc.stop()


#Tensor model
print("Tensor model\n")

tracemalloc.start()

times = 0
for i in range(5):
    #Загрузка модели
    timest = time.time()
    #model_trt = torch2trt(model, [x1], max_workspace_size=1<<30, use_onnx=True) # Эти строчки вместо двух следующих при первом запуске для сохранения модели
    #torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
    cur_times = time.time()-timest
    print("Load time TRT model {}".format(cur_times))
    times += cur_times
print("Mean load time {}\n".format(times/5))

times = 0
for i in range(5):
    #Запуск модели
    timest = time.time()
    output = model_trt(x1)
    cur_times = time.time()-timest
    print("Time to predict: {}".format(cur_times))
    times += cur_times
    #Преобразование результатов в проценты и сортировка их
    _, indices = torch.sort(output, descending = True)
    percentage = F.softmax(output, dim=1)[0] * 100.0

    #Вывод результатов в консоль
    print("{} most probable classes for {}:".format(CLASSES, namespace.file))
    for idx in indices[0][:5]:
        print('{}: {:.3f}%'.format(classes[idx], percentage[idx].item()))
    print("")
print("Mean predict time {}\n".format(times/5))

#Вывод максимума используемой памяти 
print('Peak of used memory {}\n'.format(tracemalloc.get_traced_memory()[1]))
tracemalloc.stop()
