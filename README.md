## Moving shapes dataset

A simple 2d temporal data generator written in PyTorch. Generates frames with a configurable amount of shapes of different shapes and velocities.


### Example usage:
```Python
import torchvision

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

shapes = MovingShapes(
    box_size = 16,
    surface_size = 32,
    object_size = 3,
    batch_size = 3,
    num_shapes = 3,
    speed_int = (1,3),
    device=device)

for i in range(20):
    surface = shapes.step()
    torchvision.transforms.ToPILImage()(surface[0].detach().cpu().view(32,32)).show()
```

#### Outputs:

![Image](example%20images/objs1.jpg?raw=true)
![Image](example%20images/objs2.jpg?raw=true)
![Image](example%20images/objs3.jpg?raw=true)
![Image](example%20images/objs4.jpg?raw=true)
![Image](example%20images/objs5.jpg?raw=true)
![Image](example%20images/objs6.jpg?raw=true)
![Image](example%20images/objs7.jpg?raw=true)
![Image](example%20images/objs8.jpg?raw=true)
![Image](example%20images/objs9.jpg?raw=true)
![Image](example%20images/objs10.jpg?raw=true)

### Parameters

- `surface_size` is the size of the surface the shapes are drawn on.
- `box_size` determines the boundaries in which the random vertex positions of the shapes are chosen. 
- `object_size` is the amount of lines, or rather vertices, each shape has.
- `batch_size` is the amount of examples are calculated at once. For efficient processing on GPU's, use a high batch size and divide it up in smaller batches after.
- `num_objects` determines the amount of shapes drawn on the surface in each example.
- `speed_int` is the interval from which uniformly the velocity for each shape is chosen. A speed of 1 means it moves 1 pixel per frame.

