import torch
from typing import Tuple
from PIL import Image
from PIL.Image import Resampling
import random
# import array

# TODO: make compatible with hardware acceleration
class Polygon:
	def __init__(self, 
				 points=None,
				 num_points=3,
				 size=[16,16],
				 theta=0,
				 position=None,
				 velocity=None,
				 speed_range=[0,8],
				 color=255,
				 draw_mode='lines'):
		
		if points is None:
			self.points = [[random.randint(0, size[0]), random.randint(0, size[1])] for i in range(0, num_points)]
			self.size = size
		else:
			min_pt_x = points[0][0]
			min_pt_y = points[0][1]
			max_pt_x = points[0][0]
			max_pt_y = points[0][1]
		
			for i in range(1, len(points)):
				min_pt_x = min(points[i][0], min_pt_x)
				min_pt_y = min(points[i][1], min_pt_y)
				max_pt_x = max(points[i][0], max_pt_x)
				max_pt_y = max(points[i][1], max_pt_y)
				
			self.size = [max_pt_x - min_pt_x, max_pt_y - min_pt_y]
			self.points = points
			
			for i in range(0, len(points)):
				self.points[i] = [points[i][0]-min_pt_x, points[i][1]-min_pt_y]
			
		
		self.theta = theta
		self.position = position
		self.center = [0, 0]
		
		if velocity is None:
			unit_vec = torch.randn(2, device='cpu')
			t = torch.pow(unit_vec,2)
			unit_vec = unit_vec / torch.sqrt(t + t.roll(1,t.dim()-1))
			self.velocity = unit_vec * ((torch.rand(1, device='cpu') * speed_range[1]) + speed_range[0])
			# convert this: [[random.randint(0, size[0]), random.randint(0, size[1])] for i in range(0, num_points)]
		else:
			self.velocity = velocity
		
		self.color = color
		self.base_img = Image.new('L', self.size) # grayscale
		
		if draw_mode == 'lines':
			self.draw_lines()
		elif draw_mode == 'solid':
			self.draw_solid()
		elif draw_mode == 'diff':
			self.draw_diff()
		else:
			self.draw_lines()
	
	def draw(self, img):
		tmp = self.base_img.rotate(self.theta, resample=Resampling.BILINEAR, center=self.center, expand=1, fillcolor=0)
		img.paste(tmp, box=(int(self.position[0] - self.center[0]), int(self.position[1] - self.center[1])), mask=tmp)
	
	# incorrect; needs to be fixed
	def draw_lines(self):
		lines = [0]*2
		lines[0] = self.points
		lines[1] = [*self.points[1:], self.points[0]]
		
		pixels = [0]*(self.size[0]*self.size[1])
		x_sum = 0
		y_sum = 0
		mass = 0
		
		# TODO: optimize
		for i in range(0, len(self.points)):
			v = lines[0][i]
			w = lines[1][i]
			l2 = ((v[0]-w[0])**2)+((v[1]-w[1])**2)
			
			for y in range(0, self.size[1]):
				for x in range(0, self.size[0]):
					t = (((x - v[0]) * (w[0] - v[0])) / l2, ((y - v[1]) * (w[1] - v[1])) / l2)
					t = min(max(t, (0,0)), (1,1))

					# d = dist(p, (v + t * (w - v)))
					d = (x-(v[0]+t[0]*(w[0]-v[0])))**2 + (y-(v[1]+t[1]*(w[1]-v[1])))**2
					
					# print("(" + str(x) + ", " + str(y) + "): " + str(d))
					
					if d < 1: # don't need sqrt since sqrt(d) < 1 if d < 1
						pixels[self.size[0]*y+x] = self.color
						x_sum += x
						y_sum += y
						mass += 1
						
		self.center = (x_sum / mass, y_sum / mass)
		self.base_img.putdata(pixels)

	def draw_solid(self):
		lines = [0]*2
		lines[0] = self.points
		lines[1] = [*self.points[1:], self.points[0]]
		
		pixels = [0]*(self.size[0]*self.size[1])
		x_sum = 0
		y_sum = 0
		mass = 0
		
		for y in range(0, self.size[1]):
			intersections = []
		
			for i in range(0, len(self.points)):
				if y > lines[0][i][1] and y < lines[1][i][1] or y < lines[0][i][1] and y > lines[1][i][1]:
					intersections = intersections + [int((y - lines[0][i][1]) \
						* (lines[0][i][0] - lines[1][i][0]) / (lines[0][i][1] - lines[1][i][1]) + lines[0][i][0] + 0.5)]
			
			intersections = sorted(intersections)
			x = 0
			fill = 0
		
			for i in range(0, len(intersections)):
				while x < intersections[i]:
					if fill != 0:
						x_sum += x
						y_sum += y
						mass += 1
						
					pixels[self.size[0]*y+x] = fill
					x += 1
					
				if fill == 0:
					fill = self.color
				else:
					fill = 0
				
			while x < self.size[0]:
				pixels[self.size[0]*y+x] = 0
				x += 1
						
		self.center = (x_sum / mass, y_sum / mass)
		self.base_img.putdata(pixels)
		
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

class ImageGenerator:
	def __init__(self,
				 shapes,
				 size=(128,96)):
		
		self.size = size
		self.shapes = shapes
		self.img = Image.new('L', (size[0], size[1]))
		
		
	def draw(self):
		self.img.putdata([0] * (self.size[0] * self.size[1]))
		
		for p in self.shapes:
			p.draw(self.img)			
			
			
	def move(self):
		for i in range(len(self.shapes)):
			# move the object
			self.shapes[i].position[0] += self.shapes[i].velocity[0]
			self.shapes[i].position[1] += self.shapes[i].velocity[1]

			max_ = (0, 0)
			min_ = (self.size[0], self.size[1])
			
			# calculate max and min for each batch and axis
			for j in range(0, len(self.shapes[i].points)):
				max_ = (max(max_[0], self.shapes[i].points[j][0]), max(max_[1], self.shapes[i].points[j][1]))
				min_ = (min(min_[0], self.shapes[i].points[j][0]), min(min_[1], self.shapes[i].points[j][1]))

			# check boundary violations
			exceed_right = max_[0] > self.size[0]
			exceed_left  = min_[0] < 0
			exceed_up    = max_[1] > self.size[1]
			exceed_down  = min_[1] < 0

			# calculate the exceeded distance to the boundary we have to adjust
			adjust = [0, 0]

			adjust[0] += exceed_right * (self.size[0] - max_[0])
			adjust[0] += exceed_left  * (-1. * min_[0])
			adjust[1] += exceed_up    * (self.size[1] - max_[1])
			adjust[1] += exceed_down  * (-1. * min_[1])

			# note that we have to adjust twice the exceeded distance
			self.shapes[i].position[0] += 2 * adjust[0]
			self.shapes[i].position[1] += 2 * adjust[1]
			
	
class MovingShapes:
	def __init__(self,
				 num_shapes:int,
				 object_size:int,
				 box_size:int,
				 surface_size:int,
				 batch_size:int,
				 speed_int:Tuple[int, int],
				 device = 'cpu',
				 draw_mode = 'lines'):
		self.num_shapes = num_shapes
		self.object_size = object_size
		self.num_lines = num_shapes * object_size
		self.box_size = box_size
		self.surface_size = surface_size
		self.batch_size = batch_size
		self.device = device
		self.draw_mode = draw_mode

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
		if self.draw_mode == 'lines':
			return draw_lines(self.lines, self.surface_size, device=self.device)
		elif self.draw_mode == 'solid':
			return draw_solid(self.lines, self.surface_size, device=self.device)
		else:
			return torch.zeros(2, self.surface_size*self.surface_size, device=self.device)

	def step(self):
		surface = self.draw()
		self.move()
		return surface



if __name__ == '__main__':
	import torchvision
	import argparse
	import os
	import pathlib
	
	# parse command-line arguments
	parser = argparse.ArgumentParser(
						prog = 'Moving Objects',
						description = 'Generates frames of moving objects.',
						epilog = 'This tool is open source: https://github.com/jkentw/moving-shapes')
						
	parser.add_argument('-o', '--output-path', 
						type=pathlib.Path, 
						help='Where to write each frame image.')
	parser.add_argument('-n', '--num-frames', 
						type=int, default=10, 
						help='Number of frames to generate.')
	parser.add_argument('-m', '--draw-mode', 
						choices=['lines', 'solid', 'diff'], 
						default='lines',
						help='''The way shapes are drawn. \n
							 lines: shows the outline of each shape. \n
							 solid: fills each shape. \n
							 diff: same as \'solid\', but shows the normalized difference between consecutive frames
							 ''') # TODO: newlines do not display; fix them
	
	args = parser.parse_args()
	print(args)
	
	# if applicable, create output directory
	if args.output_path is not None:
		write_to_file = True
		if not os.path.exists(args.output_path.name):
			os.makedirs(args.output_path)
			
		output_path = args.output_path.as_posix()
		
	else:
		write_to_file = False
		output_path = None
	
	# use hardware acceleration if possible
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'

	# generator parameters
	shapes = MovingShapes(
		box_size = 16,
		surface_size = 128,
		object_size = 3,
		batch_size = 3,
		num_shapes = 3,
		speed_int = (1,3),
		device=device,
		draw_mode=args.draw_mode)
	
	poly1 = Polygon(theta=0, points=[[0,0],[16,16],[8,0]], velocity=[10,6], position=[64, 32],   draw_mode=args.draw_mode)
	poly2 = Polygon(theta=0, points=[[0,0],[24,0],[4,24],[20,24]], velocity=[-8,10], position=[128, 20], draw_mode=args.draw_mode)
	
	gen = ImageGenerator(shapes=[poly1, poly2], size=[256,192])
	
	# run generator
	for i in range(args.num_frames):
		gen.draw()
		img = gen.img
		gen.move()
		gen.shapes[0].theta += 10
		gen.shapes[1].theta += -15
		print(gen.shapes[0].center)
		print(gen.shapes[0].position)
		print(gen.shapes[1].center)
		print(gen.shapes[1].position)
		
		if write_to_file:
			img.save(output_path + '/frame' + str(i) + '.png')
		else:
			img.show()
	