import json
import sys
import random


def detokenize(tokens):
    return ' '.join(tokens)


def getTasks(f, size=32, n=5, k=3):
    whole_division = json.load(open(f, encoding='utf8'))
    relations = whole_division.keys()
    print(relations)
    tasks = []
    for _ in range(size):
        task_train = []
        task_test = []
        n_relations = random.sample(relations, n)  # 随机采集 N 个关系
        print('n-rel:  ', n_relations)
        all = [random.sample(whole_division[relation], k+1) for relation in n_relations]  # 每个关系随机采集 K 个样本
        print(all[0][0])

        allTasks = []
        for rel_samps in all:
            relSamps = []
            for samp in rel_samps:
                text = samp['tokens']

                for indx in range(len(text)):
                    text[indx] = text[indx].lower()

                h = samp['h'][2][0]   # [[26, 27]]
                t = samp['t'][2][0]
                # print(text)
                # print(h)
                # print(t)
                relSamps.append((text, h, t))
            allTasks.append(relSamps)

        for i in range(len(n_relations)):
            rel = n_relations[i]
            print('rel:  ', rel)
            task_test.append((allTasks[i][0], rel))
            for j in allTasks[i][1:]:
                task_train.append((j, rel))

        random.shuffle(task_train)
        random.shuffle(task_test)

        task = {'train': task_train, 'test': task_test, 'labels': n_relations}

        tasks.append(task)

    return tasks

# getTasks('./train.json', size=1)