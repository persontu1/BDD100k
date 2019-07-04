import json

# 这里是我需要的9个类别
categorys = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']


def parseJson(jsonFile):
    '''
      params:
        jsonFile -- BDD00K数据集的一个json标签文件
      return:
        返回一个列表的列表，存储了一个json文件里面的方框坐标及其所属的类，
    '''
    objs = []
    obj = []
    info = jsonFile
    name = info['name']
    objects = info['labels']
    for i in objects:
        if (i['category'] in categorys):
            obj.append(int(i['box2d']['x1']))
            obj.append(int(i['box2d']['y1']))
            obj.append(int(i['box2d']['x2']))
            obj.append(int(i['box2d']['y2']))
            obj.append(i['category'])
            objs.append(obj)
            obj = []
    # print("objs",objs)
    return name, objs


# test
file_handle = open('/dataset/bdd1/label/traindrive_val.txt', mode='a')
f = open("/dataset/bdd1/label/bdd100k/labels/bdd100k_labels_images_val.json")
info = json.load(f)
objects = info
n = len(objects)
for i in range(n):
    an = ""
    name, result = parseJson(objects[i])
    an = "/dataset/bdd1/val/" + name
    for j in range(len(result)):
        cls_id = categorys.index(result[j][4])
        an = an + ' ' + str(result[j][0]) + ',' + str(result[j][1]) + ',' + str(result[j][2]) + ',' + str(
            result[j][3]) + ',' + str(cls_id)
    an = an + '\n'
    file_handle.write(an)
    print(len(result))
    print(an)
