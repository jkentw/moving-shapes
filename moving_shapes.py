
import torch
from typing import Tuple

def dist(v, w):
    '''calculates distance between points 'v' and 'w'.
    last dimension is assumed to be size 2
    '''
    return torch.sum(torch.pow(v-w,2), dim=v.dim()-1)

def indices(dim, device='cpu'):
    '''returns indices for a matrix with dimensions 'dim' '''
    p = torch.zeros((2,dim[0]*dim[1]), device=device)
    p[0] = torch.arange(dim[0], device=device).repeat(dim[1])
    p[1] = torch.repeat_interleave(torch.arange(dim[1], device=device), dim[0])
    return p

def draw_lines(line, surface_size, device='cpu'):
    '''Calculates the distance of all points in a box of given size
    to a certain line segment and uses the negative of this to shade pixels
    and so draws a line segment.

    Inspiration from: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    '''
    num_lines = line.shape[1]
    batch_size = line.shape[2]

    v = line[0]
    w = line[1]

    p = indices((surface_size,surface_size), device=device).t().expand(num_lines, batch_size, surface_size*surface_size, 2).transpose(1,2).transpose(0,1)

    l2 = dist(v,w)

    l2 = l2.expand(2, *l2.shape).transpose(0,2).transpose(0,1)

    t = ((p - v) * (w - v)) / l2
    t = t + t.roll(1,t.dim()-1)
    t = torch.clamp(t, 0., 1.)

    d = dist(p, (v + t * (w - v)))
    d = torch.clamp(d, 0., 1.)
    d = torch.sum(1. - d, dim=1).clamp(0., 1.)
    return d.t()



def create_shape(num_points=3, box_size=16, batch_size=32, device='cpu'):
    '''creates a set of lines between a number of random points contained
    within a box of size 'box_size'
    '''
    dims = 2 # 2d shapes, no others necessary

    points = torch.rand((num_points,batch_size,dims), device=device)*box_size
    lines = torch.zeros((2,num_points,batch_size,dims), device=device)
    lines[0] = points
    lines[1] = points.roll(1,0)
    return lines

def rand_unit_vec(dim, device='cpu'):
    '''returns a unit vector of a random direction'''
    a = torch.randn((*dim,2), device=device)
    t = torch.pow(a,2)
    return a / torch.sqrt(t + t.roll(1,t.dim()-1))

class MovingShapes:

    def __init__(self,
                 num_shapes:int,
                 object_size:int,
                 box_size:int,
                 surface_size:int,
                 batch_size:int,
                 speed_int:Tuple[int, int],
                 device = 'cpu'):
        self.num_shapes = num_shapes
        self.object_size = object_size
        self.num_lines = num_shapes * object_size
        self.box_size = box_size
        self.surface_size = surface_size
        self.batch_size = batch_size
        self.device = device

        # store all the lines of the shapes in one place so they can be drawn in parallel
        self.lines = torch.zeros((2,self.num_lines,batch_size,2), device=device)
        for i in range(num_shapes):
            # index for the lines for this object
            idx = range(i*self.object_size,(i+1)*self.object_size)
            # generate random shape
            self.lines[:,idx] = create_shape(num_points=object_size,
                                             box_size=box_size,
                                             batch_size=batch_size,
                                             device=device)

        # offset the box at a random position on the surface
        start_position = torch.rand((num_shapes, batch_size, 2), device=device) * (surface_size - box_size)
        start_position = torch.repeat_interleave(start_position, object_size, dim=0)
        self.lines = self.lines + start_position

        # random starting movement direction and speed
        # kept separately so we can dial them separately during trials
        self.directions = rand_unit_vec((num_shapes, batch_size), device=device)
        self.speeds = (torch.rand((num_shapes, batch_size), device=device) * speed_int[1]) + speed_int[0]

    def move(self):
        '''moves object in direction of the unit vector 'direction'
        with magnitude 'speed'

        if the surface boundary is exceeded, object position and
        direction is adjusted
        '''

        # we have to check each object separately
        for i in range(self.num_shapes):
            # index for the lines for this object
            idx = range(i*self.object_size,(i+1)*self.object_size)

            # move the object
            self.lines[:,idx] += (self.directions[i].t() * self.speeds[i]).t()

            # calculate max and min for each batch and axis
            max_ = self.lines[:,idx].max(dim=0).values.max(dim=0).values.t()
            min_ = self.lines[:,idx].min(dim=0).values.min(dim=0).values.t()

            # check boundary violations
            exceed_right = max_[0] > self.surface_size
            exceed_left  = min_[0] < 0.
            exceed_up    = max_[1] > self.surface_size
            exceed_down  = min_[1] < 0.

            # calculate the exceeded distance to the boundary we have to adjust
            adjust = torch.zeros(2, self.batch_size, device=self.device)

            adjust[0] += exceed_right * (self.surface_size - max_[0])
            adjust[0] += exceed_left  * (-1. * min_[0])
            adjust[1] += exceed_up    * (self.surface_size - max_[1])
            adjust[1] += exceed_down  * (-1. * min_[1])

            # note that we have to adjust twice the exceeded distance
            self.lines[:,idx] += 2. * adjust.t()

            # adjust direction of movement
            self.directions[i,:,0] *= (~(exceed_right | exceed_left) * 2 - 1)
            self.directions[i,:,1] *= (~(exceed_up    | exceed_down) * 2 - 1)

        return self.lines

    def draw(self):
        return draw_lines(self.lines, self.surface_size, device=self.device)

    def step(self):
        surface = self.draw()
        self.move()
        return surface




if __name__ == '__main__':
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

    for i in range(10):
        surface = shapes.step()
        torchvision.transforms.ToPILImage()(surface[0].detach().cpu().view(32,32)).show()
