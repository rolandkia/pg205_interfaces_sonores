import threading

import pygame
import pygame.gfxdraw
import numpy as np
import pyaudio

import pygame_gui
from pygame_gui.windows.ui_file_dialog import UIFileDialog
from pygame.rect import Rect

from styleRecognition import predict_song_for_graphics
from styleRecognition import predict_song_from_mic

from styleRecognition import nb_features
from styleRecognition import genres

import sounddevice as sd
import soundfile as sf


def smooth_interpolation(new, prev, alpha=0.01):
	return (1 - alpha) * prev + alpha * new

def reRange(OldValue, OldMin, OldMax, NewMin, NewMax):
	OldRange = (OldMax - OldMin)
	if (OldRange == 0):
		NewValue = NewMin
	else :
		NewRange = (NewMax - NewMin)  
		NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
	return NewValue


class RectButton():
	def __init__(self, x, y, w, h, path):
		self.image = pygame.transform.scale(pygame.image.load(path), (w, h))
		self.rect = pygame.Rect(x, y, w, h)
		self.x, self.y = x, y
		self.color_normal = (225, 225, 225)
		self.color_survol = (180, 190, 175)
		self.color_clic = (160, 170, 155)
		self.current_color = self.color_normal

	def update_survol_color(self):
		self.current_color = self.color_survol

	def is_inside(self, mx, my):
		return self.rect.collidepoint(mx, my)

	def display(self, surface):
		pygame.gfxdraw.box(surface, self.rect, self.current_color)
		border_color = [max(c - 15, 0) for c in self.current_color]
		pygame.gfxdraw.rectangle(surface, self.rect, border_color)
		surface.blit(self.image, (self.x, self.y))
		self.current_color = self.color_normal

class CircleButton():
	def __init__(self, x, y, r, path):
		self.image = pygame.transform.scale(pygame.image.load(path), (2*r, 2*r))
		self.x, self.y, self.r = int(x), int(y), int(r)
		self.color_normal = (225, 225, 225)
		self.color_survol = (180, 190, 175)
		self.color_clic = (160, 170, 155)
		self.current_color = self.color_normal
		self.is_one = False

	def turn_off_one(self):
		self.is_one = not self.is_one

	def update_survol_color(self):
		self.current_color = self.color_survol

	def is_inside(self, mx, my):
		return (mx - self.x)**2 + (my - self.y)**2 < self.r**2

	def display(self, surface):
		pygame.gfxdraw.filled_circle(surface, self.x, self.y, self.r, self.current_color)
		border_color = [max(c - 15, 0) for c in self.current_color]
		pygame.gfxdraw.aacircle(surface, self.x, self.y, self.r, border_color)
		surface.blit(self.image, (self.x - self.r, self.y - self.r))
		if not self.is_one:
			self.current_color = self.color_normal

class Label():
	def __init__(self, x, y, text_x, text_y, name, color):
		self.x = x
		self.y = y
		self.name = name
		self.color = color
		self.r = 5

		self.font = pygame.font.Font(None, 18) 
		self.text_x = text_x
		self.text_y = text_y
	
	def update(self, d, c_x, c_y):
		vector = np.array([self.text_x, self.text_y])
		norm = np.linalg.norm(vector)
		if norm != 0:
			vector /= norm
			
		self.x = smooth_interpolation(c_x + vector[0]*d, self.x, alpha=0.05)
		self.y = smooth_interpolation(c_y + vector[1]*d, self.y, alpha=0.05)

	
	def display(self, surface):
		
		x, y = int(self.x), int(self.y)
		pygame.gfxdraw.filled_circle(surface, x, y, self.r, self.color)
		text_surface = self.font.render(self.name, True, (0, 0, 0)) 
		text_rect = text_surface.get_rect(center=(x + self.text_x, y + self.text_y))
		surface.blit(text_surface, text_rect)

class Animation():
	def __init__(self, x, y, max_r, g = genres):
		self.x = x
		self.y = y
		self.r = 0
		self.max_r = max_r
		self.genres = g
		self.labels = []
		for i in range(len(g)):
			a = i-1
			tmp_x = x + self.max_r*np.cos(2*a*np.pi/len(g))
			tmp_y = y + self.max_r*np.sin(2*a*np.pi/len(g))
			tmp_vector = np.array([tmp_x - x, tmp_y - y])
			tmp_vector = 20*tmp_vector/np.linalg.norm(tmp_vector)
			
			self.labels.append(Label(x, y, tmp_vector[0], tmp_vector[1],self.genres[i], (50*i, 0, 0)))
		
	def update(self, inc_r):
		# if self.r < self.max_r:
		#     self.r += inc_r
		#     for i in range(len(self.labels)):
		#         a = i-1
		#         x = self.x + self.r*np.cos(2*a*np.pi/len(self.labels))
		#         y = self.y + self.r*np.sin(2*a*np.pi/len(self.labels))
		#         self.labels[i].update(x, y)
		# for l in self.labels:
		#     l.update()
		pass
		
	def display(self, surface, ai_res = False):
		pygame.gfxdraw.circle(surface, int(self.x), int(self.y), int(self.max_r), (0, 0, 0))
		
		pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), 3, (40, 40,40))

		if ai_res:
			self.font = pygame.font.Font(None, 18) 
			for i in range(len(ai_res)):
				a = i-1
				proba = int(ai_res[a]*100)
				text_surface = self.font.render(str(proba) + " %", True, (0, 0, 0)) 
				vector = np.array([self.labels[a].text_x, self.labels[a].text_y])
				norm = np.linalg.norm(vector)
				if norm != 0:
					vector /= norm
				x = self.x + vector[0]*(self.max_r+ 10)
				y = self.y + vector[1]*(self.max_r+ 10)
				text_rect = text_surface.get_rect(center=(x, y))
				surface.blit(text_surface, text_rect)
				
		for label in self.labels:
			label.display(surface)
		
		for i in range(len(self.labels)):
			x1 = int(self.labels[i].x)
			y1 = int(self.labels[i].y)
			if i+1 < len(self.labels):
				x2 = int(self.labels[i+1].x)
				y2 = int(self.labels[i+1].y)
			else:
				x2 = int(self.labels[0].x)
				y2 = int(self.labels[0].y)
			pygame.gfxdraw.line(surface, x1, y1, x2, y2, (0, 0, 0))
		

is_file_selected = False
selected_file = ""
frequency = 2
total_duration = 30
nb_time = 0
nb_samples = 1 
running = True
ai_res = []
model_used = 'with_contrast'
nb_mfcc = 10
nb_features = 23 - (20 - nb_mfcc)
sum_adjustement = 0
if (model_used == 'with_contrast'):
    nb_features += 1
if (model_used == 'without_zcr_tempo'):
    nb_features -= 2
    sum_adjustement = 1
s_features = [0 for i in range(nb_features - 1 + sum_adjustement)]
is_mic_one = False
data = []

def main_loop():
	global selected_file, is_file_selected, running, ai_res, nb_time, nb_samples, s_features, is_mic_one, data
	
	pygame.init()
	pygame.mixer.init()

	# --- Paramètres de base ---
	WIDTH, HEIGHT = 600, 800
	BACKGROUND_COLOR = (150, 150, 150)
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("")
	clock = pygame.time.Clock()

	# --- Gestionnaire pygame_gui ---
	manager = pygame_gui.UIManager((WIDTH, HEIGHT))  # tu peux ajouter un thème ici si tu veux
	file_dialog = None

	# --- PyAudio ---
	CHUNK = 1024 # factor of 3675 use value for ai frequency is 6
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 22050
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

	# --- Boutons ---
	b = RectButton(WIDTH - 40 - 15, 15, 40, 40, "./file.png")
	mic = CircleButton(WIDTH/2, 100, 40, "./l.png")

	anim = Animation(WIDTH/2, HEIGHT-280, 250)

	prev_radius = 35
	while running:

		time_delta = clock.tick(60) / 1000.0
		screen.fill(BACKGROUND_COLOR)

		data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

		# Convertir en float
		data = data.astype(np.float32)

		# Calculer la valeur absolue maximale pour normaliser entre -1 et 1
		max_val = np.max(np.abs(data))
		if max_val == 0:
			normalized_data = data  # rien à normaliser
		else:
			normalized_data = data / max_val
			
		
		fft_vals = np.abs(np.fft.rfft(data))
		bass_energy = 0
		if np.max(fft_vals) != 0:
			fft_data = fft_vals / np.max(fft_vals)
			bass_energy = np.mean(fft_data[:120])

		data = normalized_data.copy()

		mx, my = pygame.mouse.get_pos()
		if b.is_inside(mx, my): b.update_survol_color()
		if mic.is_inside(mx, my): mic.update_survol_color()

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

			elif event.type == pygame.MOUSEBUTTONDOWN:
				if mic.is_inside(mx, my):
					mic.turn_off_one()
				if b.is_inside(mx, my) and not file_dialog:
					is_mic_one = False
					mic.is_one = False
					dialog_width, dialog_height = 500, 400
					dialog_x = 0
					dialog_y = 0
					file_dialog = UIFileDialog(
						rect=pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height),
						manager=manager,
						allow_picking_directories=False
					)

			elif event.type == pygame.USEREVENT:
				if event.user_type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
					s_features = [0 for i in range(24-1)]
					nb_time = 0
					nb_samples = 1 
					selected_file = event.text
					pygame.mixer.music.stop() 
					pygame.mixer.music.load(selected_file)
					pygame.mixer.music.play(loops=-1)  # Jouer en boucle>
					is_file_selected = True
					file_dialog = None
					is_mic_one = False
				elif event.user_type == pygame_gui.UI_WINDOW_CLOSE and event.ui_element == file_dialog:
					file_dialog = None
										

			manager.process_events(event)

		if mic.is_one:
			is_file_selected = False
			is_mic_one = True
			pygame.mixer.music.stop() 
			target_radius = min(35 + bass_energy * 400, 100)
			prev_radius = smooth_interpolation(target_radius, prev_radius, 0.05)
			pygame.gfxdraw.aacircle(screen, WIDTH//2, mic.y, int(prev_radius), (200, 0, 0))
			pygame.gfxdraw.filled_circle(screen, WIDTH//2, mic.y, int(prev_radius), (200, 0, 0))
	
		if not mic.is_one:
			is_mic_one = False

	
		if (is_file_selected or is_mic_one )and len(ai_res) == 5:
		
			for a in range(len(anim.labels)):
				i = a - 1
				d = reRange(ai_res[i], 0, 1, 0, anim.max_r)
				anim.labels[i].update(d, anim.x, anim.y)
		
	
		b.display(screen)
		mic.display(screen)
		anim.display(screen, ai_res=ai_res)

		manager.update(time_delta)
		manager.draw_ui(screen)

		pygame.display.flip()
				   
	stream.stop_stream()
	stream.close()
	p.terminate()
	pygame.quit()

def ai_loop():
	global is_file_selected, nb_samples, nb_time, total_duration, frequency, selected_file, running, ai_res, s_features, is_mic_one, data, model_used, nb_mfcc
	
	while running:
		if not is_mic_one:
			if is_file_selected and nb_time < total_duration: 
				ai_res, s_features = predict_song_for_graphics(selected_file, model_used, nb_mfcc, s_features ,nb_time, nb_samples)
				nb_time += 1/frequency
				nb_samples += 1
			else:
				ai_res = []
				nb_features = 23 - (20 - nb_mfcc)
				sum_adjustement = 0
				if (model_used == 'with_contrast'):
					nb_features += 1
				if (model_used == 'without_zcr_tempo'):
					nb_features -= 2
					sum_adjustement = 1
				s_features = [0 for i in range(nb_features - 1 + sum_adjustement)]
		elif is_mic_one:
			ai_res, s_features = predict_song_from_mic(data, model_used, nb_mfcc, s_features, nb_samples)
			nb_samples += 1
  

thread1 = threading.Thread(target=ai_loop)
thread2 = threading.Thread(target=main_loop)

thread1.start()
thread2.start()

thread1.join()
thread2.join()



