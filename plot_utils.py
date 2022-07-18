import matplotlib.pyplot as plt
from graphviz import Digraph

def train_curves(x_history, model_name):
    loss = x_history["train_loss"]
    val_loss = x_history["val_loss"]

    type_acc = x_history["train_typ_accs"]
    val_type_acc = x_history["val_typ_accs"]

    loc_acc = x_history["train_loc_accs"]
    val_loc_acc = x_history["val_loc_accs"]

    plt.figure(figsize=(10, 8))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Loss")
    plt.ylim([0,1])
    plt.title(model_name + " Loss Curves")
    plt.show()
    print()

    plt.figure(figsize=(10, 8))
    plt.plot(type_acc, label="Training Type Accuracy")
    plt.plot(val_type_acc, label="Validation Type Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Type Accuracy")
    plt.ylim([0.5,1])
    plt.title(model_name + " Type Accuracy")
    plt.show()
    print()
    
    plt.figure(figsize=(10, 8))
    plt.plot(loc_acc, label="Training Location Accuracy")
    plt.plot(val_loc_acc, label="Validation Location Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Location Accuracy")
    plt.ylim([0.5,1])
    plt.title(model_name + " Location Accuracy")
    plt.show()
    print()



def plot(genotype, filename, epoch):
    g = Digraph(format='png', 
                graph_attr=dict(size =  '7.0, 5.0', bgcolor="white", ratio="fill"),
                edge_attr=dict(fontsize='18', fontname="times"),
                node_attr=dict(style='filled', shape='rect', align='center',
                               fontsize='20', height='0.5', width='0.5',
                               penwidth='2', fontname="times"),
                engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    # for i in range(steps):
    g.edge(str(steps-1), "c_{k}", fillcolor="gray")
    g.edge(str(steps-2), "c_{k}", fillcolor="gray")

    g.node("Epoch " + epoch)

    g.render(filename + epoch)