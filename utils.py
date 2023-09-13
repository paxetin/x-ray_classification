import matplotlib.pyplot as plt


def form_str(e, t):
    e_str = str(e)
    t_str = str(t)
    num_blanks = len(t_str) - len(e_str)
    return ' ' * num_blanks + e_str if num_blanks > 0 else e_str

class Summary:
    def __init__(self, train_loader, val_loader):
        self.lengths = {'train_batch_amounts': len(train_loader),
                        'train_dataset_size': len(train_loader.dataset),
                        'val_batch_amounts': len(val_loader),
                        'val_dataset_size': len(val_loader.dataset)}

        self.loss_acc = {'train_loss':[],
                         'val_loss': [],
                         'train_acc': [],
                         'val_acc': []}
    
    def compute(self, batch_idx, loss, outputs, labels, mode=None):
        if batch_idx == 0:
            self.loss_acc[f'{mode}_loss'].append(0)
            self.loss_acc[f'{mode}_acc'].append(0)

        self.loss_acc[f'{mode}_loss'][-1] += loss.cpu().item()
        self.loss_acc[f'{mode}_acc'][-1] += (outputs.argmax(1) == labels).sum().cpu().item()

        if batch_idx == self.lengths[f'{mode}_batch_amounts'] - 1:
            self.loss_acc[f'{mode}_loss'][-1] /= self.lengths[f'{mode}_batch_amounts']
            self.loss_acc[f'{mode}_acc'][-1] /= self.lengths[f'{mode}_dataset_size']

    def visualize(self):
        if len(self.loss_acc[f'train_loss']) == 0:
            raise Exception('No data is stored.')
        else:
            for target in ['loss', 'acc']:
                plt.plot(self.loss_acc[f'train_{target}'], label=f'train {target}')
                plt.plot(self.loss_acc[f'val_{target}'], label=f'val {target}')
                plt.legend()
                plt.show()

    @property
    def train_loss(self):
        return self.loss_acc['train_loss'][-1]
    
    @property
    def train_acc(self):
        return self.loss_acc['train_acc'][-1]
    
    @property
    def val_loss(self):
        return self.loss_acc['val_loss'][-1]
    
    @property
    def val_acc(self):
        return self.loss_acc['val_acc'][-1]