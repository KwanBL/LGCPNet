from glob import *
import numpy as np
import os
import pymesh
import shutil
import random

if __name__ == '__main__':

	root = '/home/bailiang/data/labeled_meshes/SHAPENET_MESHES'
	new_root = '/home/bailiang/codes/guanbiliang/3d_seg_Pistol/test2500/'
	npoint = 1500

	category_ids = {
		'Airplane': '02691156',
		'Bag': '02773838',
		'Cap': '02954340',
		'Car': '02958343',
		'Chair': '03001627',
		'Earphone': '03261776',
		'Guitar': '03467517',
		'Knife': '03624134',
		'Lamp': '03636649',
		'Laptop': '03642806',
		'Motorbike': '03790512',
		'Mug': '03797390',
		'Pistol': '03948459',
		'Rocket': '04099429',
		'Skateboard': '04225987',
		'Table': '04379243',
	}

	for type, id in category_ids.items():
		new_file = os.path.join(new_root, id)
		if os.path.exists(new_file):
			shutil.rmtree(new_file)
		os.mkdir(new_file)

		for phrase in ['train', 'test']:
			new_file = os.path.join(new_root, id, phrase)
			if not os.path.exists(new_file):
				os.mkdir(new_file)

			new_f = os.path.join(new_file +'/' + 'points')
			if os.path.exists(new_f):
				shutil.rmtree(new_f)
			os.mkdir(new_f)
			data_save = new_f

			new_f = os.path.join(new_file +'/' + 'points_label')
			if os.path.exists(new_f):
				shutil.rmtree(new_f)
			os.mkdir(new_f)
			label_save = new_f

			# phrase_path = os.path.join(type_path, phrase)

			phrase_path = open(os.path.join(root, phrase, type + '.txt'))
			# files = os.path.join(type_path, phrase_path, '.off')
			files = phrase_path.readlines()
			filelist = []
			for file in files:
				# load mesh
				mesh = pymesh.load_mesh(os.path.join(root, type, file[:-1] + '.off'))
				label_file = open(os.path.join(root, type, file[:-1] + '_labels.txt'))
				lines = label_file.readlines()
				face_lab = mesh.faces.copy()[:, 1]

				lab = 0
				for line in range(len(lines)):

					if len(lines[line]) > 3:

						if lines[line][-2] == 'A':
							st = lines[line + 1]
							label = st.split()
							label_num = list(map(int, label))
							for li in range(len(label_num)):
								face_lab[label_num[li] - 1] = lab+1

						if lines[line][-2] == 'B':
							st = lines[line + 1]
							label = st.split()
							label_num = list(map(int, label))
							for li in range(len(label_num)):
								face_lab[label_num[li] - 1] = lab+2

						if lines[line][-2] == 'C':
							st = lines[line + 1]
							label = st.split()
							label_num = list(map(int, label))
							for li in range(len(label_num)):
								face_lab[label_num[li] - 1] = lab+3

						if lines[line][-2] == 'D':
							st = lines[line + 1]
							label = st.split()
							label_num = list(map(int, label))
							for li in range(len(label_num)):
								face_lab[label_num[li] - 1] = lab+4

						if lines[line][-2] == 'E':
							st = lines[line + 1]
							label = st.split()
							label_num = list(map(int, label))
							for li in range(len(label_num)):
								face_lab[label_num[li] - 1] = lab+5

						if lines[line][-2] == 'F':
							st = lines[line + 1]
							label = st.split()
							label_num = list(map(int, label))
							for li in range(len(label_num)):
								face_lab[label_num[li] - 1] = lab+6

				# get elements
				vertices = mesh.vertices.copy()
				faces = mesh.faces.copy()
				length = len(face_lab)
				# move to center
				center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
				vertices -= center

				# normalize
				max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
				max_len = np.sqrt(max_len)

				vertices /= max_len

				# get normal vector
				mesh = pymesh.form_mesh(vertices, faces)
				mesh.add_attribute('face_normal')
				face_normal = mesh.get_face_attribute('face_normal')

				centers = []
				for f in faces:
					[v1, v2, v3] = f
					x1, y1, z1 = vertices[v1]
					x2, y2, z2 = vertices[v2]
					x3, y3, z3 = vertices[v3]
					centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])

				centers = np.asarray(centers)
				_, filename = os.path.split(file)
				data = np.concatenate([centers, face_normal], axis=1)
				data = data.tolist()
				face_lab = face_lab.tolist()
				index = [i for i in range(0, length - 1)]

				# if phrase is 'train':
				# 	random.shuffle(index)

				if npoint >= length:
					if npoint - length > 0:
						for i in range(npoint - length):
							data.append(data[index[i]])
							face_lab.append(face_lab[index[i]])

					data = np.asarray(data)
					face_lab = np.asarray(face_lab)

					np.savetxt(data_save + '/' + filename[:-4] + '.txt', data,
					           fmt='%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f')
					np.savetxt(label_save + '/' + filename[:-4] + '.txt', face_lab,
					           fmt="%i")

				if npoint < length:
					num = length // npoint
					rem = length % npoint
					for i in range(num):
						fill = index[npoint * i: npoint * i + npoint]
						data1 = []
						face_lab1 = []
						for j in fill:
							data1.append(data[j])
							face_lab1.append(face_lab[j])

						data1 = np.asarray(data1)
						face_lab1 = np.asarray(face_lab1)
						sti = str(i)

						np.savetxt(data_save + '/' + filename[:-4] + sti + '.txt', data1,
						           fmt='%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f')
						np.savetxt(label_save + '/' + filename[:-4] + sti + '.txt', face_lab1, fmt="%i")


					fill = index[-npoint : ]

					data2 = []
					face_lab2 = []
					for j in fill:
						data2.append(data[j])
						face_lab2.append(face_lab[j])

					data2 = np.asarray(data2)
					face_lab2 = np.asarray(face_lab2)
					# for i in range(npoint - rem):
					#     index = np.random.randint(0, length)
					#     fill_face.append(centers[index])
					#     fill_label.append(face_lab[index])
					#     fill_normal.append(face_normal[index])
					i = i + 1
					sti = str(i)
					np.savetxt(data_save + '/' + filename[:-4] + sti + '.txt', data2,
					           fmt='%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f')
					np.savetxt(label_save + '/' + filename[:-4] + sti + '.txt',
					           face_lab2, fmt="%i")


			print('XXX is done!')

			print(phrase + '_' + type + '_' + filename[:-4] + ' is done!')