from typing import Tuple
from PIL import Image
from PIL.Image import Resampling
import random

class Polygon:
	def __init__(self, 
				 points=None,
				 num_points=3,
				 size=[16,16],
				 theta=0,
				 angular_velocity=0,
				 position=None,
				 velocity=None,
				 speed_range=[0,8],
				 color=255,
				 draw_mode='lines'):
		
		if points is None:
			points = [[random.randint(0, size[0]), random.randint(0, size[1])] for i in range(0, num_points)]

		min_pt_x = points[0][0]
		min_pt_y = points[0][1]
		max_pt_x = points[0][0]
		max_pt_y = points[0][1]
	
		for i in range(1, len(points)):
			min_pt_x = min(points[i][0], min_pt_x)
			min_pt_y = min(points[i][1], min_pt_y)
			max_pt_x = max(points[i][0], max_pt_x)
			max_pt_y = max(points[i][1], max_pt_y)
			
		# size should account for possible rotation (which depends on center of mass, not geometry)
		# add margins equal to the maximum size of the shape
		# (there is probably a better way to do this but it works)
		shape_size = int(((max_pt_x - min_pt_x)**2 + (max_pt_y - min_pt_y)**2) ** 0.5 + 1)
		self.size = [max_pt_x - min_pt_x + 2*shape_size, max_pt_y - min_pt_y + 2*shape_size]
		self.points = points
		
		for i in range(0, len(points)):
			self.points[i] = [points[i][0]-min_pt_x+shape_size, points[i][1]-min_pt_y+shape_size]	
		
		self.theta = theta
		self.angular_velocity = angular_velocity
		self.position = position
		self.center = [0, 0]
		
		if velocity is None:
			pass
			# convert this: [[random.randint(0, size[0]), random.randint(0, size[1])] for i in range(0, num_points)]
		else:
			self.velocity = velocity
		
		self.color = color
		self.base_img = Image.new('L', self.size) # grayscale
		
		if draw_mode == 'lines':
			self.draw_lines()
		elif draw_mode == 'solid' or draw_mode == 'diff':
			self.draw_solid()
		else:
			self.draw_solid()
	
	def draw(self, img):
		tmp = self.base_img.rotate(self.theta, resample=Resampling.BILINEAR, center=self.center, expand=0, fillcolor=0)
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
		
		epsilon = 0.001
		
		for y in range(0, self.size[1]):
			for i in range(0, len(self.points)):
				if y + epsilon > lines[0][i][1] and y - epsilon < lines[1][i][1] or \
				y - epsilon < lines[0][i][1] and y + epsilon > lines[1][i][1]:
					if(lines[0][i][1] == lines[1][i][1]):
						for x in range(lines[0][i][0], lines[1][i][0]):
							pixels[self.size[0] * y + x] = self.color
							mass += 1
							x_sum += x
							y_sum += y
					else:
						x = int((y - lines[0][i][1]) * (lines[0][i][0] - lines[1][i][0]) /
							(lines[0][i][1] - lines[1][i][1]) + lines[0][i][0])
						pixels[self.size[0] * y + x] = self.color
						mass += 1
						x_sum += x
						y_sum += y
							
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
				if y >= lines[0][i][1] and y < lines[1][i][1] or y <= lines[0][i][1] and y > lines[1][i][1]:
					intersections = intersections + [int((y - lines[0][i][1]) \
						* (lines[0][i][0] - lines[1][i][0]) / (lines[0][i][1] - lines[1][i][1]) + lines[0][i][0] + 0.5)]
			
			intersections = sorted(intersections)
			x = 0
			fill = 0
			
			print(intersections)
		
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


class ImageGenerator:
	def __init__(self,
				 shapes,
				 draw_mode='diff',
				 size=(128,96)):
		
		self.size = size
		self.shapes = shapes
		self.draw_mode = draw_mode
		self.img = Image.new('L', (size[0], size[1]))
		
		if(draw_mode == 'diff'):
			self.last_pixels = ([0] * (self.size[0] * self.size[1]))
			self.diff_pixels = ([0] * (self.size[0] * self.size[1]))
			self.diff_img = Image.new('L', (size[0], size[1]))
			self.draw() # first call to initialize last_pixels
		
		
	def draw(self):
		self.img.putdata([0] * (self.size[0] * self.size[1]))
		
		for p in self.shapes:
			p.draw(self.img)
		
		if self.draw_mode == 'diff':
			self.curr_pixels = list(self.img.getdata())
			
			min_val = 255
			max_val = 0
			
			for i in range(len(self.diff_pixels)):
				self.diff_pixels[i] = self.curr_pixels[i] - self.last_pixels[i]
				max_val = max(max_val, self.diff_pixels[i])
				min_val = min(min_val, self.diff_pixels[i])
				
			if max_val == min_val:
				max_val = 1
				min_val = 0
			
			# normalization & deep copy
			for i in range(len(self.diff_pixels)):
				self.diff_pixels[i] = int((self.diff_pixels[i] - min_val) / (max_val - min_val) * 255)
				self.last_pixels[i] = self.curr_pixels[i]
				
			self.diff_img.putdata(self.diff_pixels)
			
			
	def move(self):
		for i in range(len(self.shapes)):
			# move the object
			self.shapes[i].position[0] += self.shapes[i].velocity[0]
			self.shapes[i].position[1] += self.shapes[i].velocity[1]
			self.shapes[i].theta += self.shapes[i].angular_velocity

			max_ = [0, 0]
			min_ = [0, 0]
			# min_ = [self.size[0], self.size[1]]
			
			# calculate max and min for each axis
			# for j in range(0, len(self.shapes[i].points)):
			#	max_ = [max(max_[0], self.shapes[i].points[j][0]), max(max_[1], self.shapes[i].points[j][1])]
			#	min_ = [min(min_[0], self.shapes[i].points[j][0]), min(min_[1], self.shapes[i].points[j][1])]	
			
			max_[0] += self.shapes[i].position[0]
			max_[1] += self.shapes[i].position[1]
			min_[0] += self.shapes[i].position[0]
			min_[1] += self.shapes[i].position[1]
			
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
			
			# flip direction if necessary
			self.shapes[i].velocity[0] *= (1-2*exceed_left)
			self.shapes[i].velocity[0] *= (1-2*exceed_right)
			self.shapes[i].velocity[1] *= (1-2*exceed_up)
			self.shapes[i].velocity[1] *= (1-2*exceed_down)
			

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
	# print(args)
	
	# if applicable, create output directory
	if args.output_path is not None:
		write_to_file = True
		if not os.path.exists(args.output_path.name):
			os.makedirs(args.output_path)
			
		output_path = args.output_path.as_posix()
		
	else:
		write_to_file = False
		output_path = None
	
	poly1 = Polygon(theta=0, angular_velocity=10, points=[[0,0],[32,32],[8,48]], velocity=[3,5], position=[64, 128], draw_mode=args.draw_mode)
	poly2 = Polygon(theta=0, angular_velocity=-5, points=[[0,0],[48,0],[8,48],[40,60]], velocity=[-6,4], position=[96, 20], draw_mode=args.draw_mode)
	poly3 = Polygon(theta=30, angular_velocity=-2, points=[[0,0],[32,0],[32,32],[0,32]], velocity=[4,-3], position=[192, 40], draw_mode=args.draw_mode)
	
	gen = ImageGenerator(shapes=[poly1, poly2, poly3], size=[256,192])
	
	# run generator
	for i in range(args.num_frames):
		gen.draw()
		
		if args.draw_mode == 'diff':
			img = gen.diff_img
		else:
			img = gen.img
		
		gen.move()
		
		print("--- frame " + str(i))
		
		for j in range(len(gen.shapes)):
			print("shape " + str(j) + ": " + str(gen.shapes[j].center[0] + gen.shapes[j].position[0]) +
				", " + str(gen.shapes[j].center[1] + gen.shapes[j].position[1]))
		
		if write_to_file:
			img.save(output_path + '/frame' + str(i) + '.png')
		else:
			img.show()
	