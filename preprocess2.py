from glob import *
import numpy as np
import os
import pymesh
import shutil
import random

if __name__ == '__main__':

	root = '/home/bailiang/codes/guanbiliang/data/labeled_meshes/PSB_COSEG_MESHES'
	list_root = '/home/bailiang/codes/MeshCNN/datasets'
	new_root = '/home/bailiang/codes/guanbiliang/3d_seg_Pistol/PSB_COSEG_test2500/'
	npoint = 2048

	category_ids = {
	# 'psbAirplane': '01',
	# 'psbAnt': '02',
	# 'psbArmadillo': '03',
	# 'psbBearing': '04',
	# 'psbBird': '05',
	# 'psbBust': '06',
	# 'psbChair': '07',
	# 'psbCup': '08',
	# 'psbFish': '09',
	# 'psbFourLeg': '10',
	# 'psbGlasses': '11',
	# 'psbHand': '12',
	# 'psbHuman': '13',
	# 'psbMech': '14',
	# 'psbOctopus': '15',
	# 'psbPlier': '16',
	# 'psbTable': '17',
	# 'psbTeddy': '18',
	# 'psbVase': '19',
	# 'cosegCandelabra': '20',
	# 'cosegChairs': '21',
	# 'cosegFourleg': '22',
	# 'cosegGoblets': '23',
	# 'cosegGuitars': '24',
	# 'cosegIrons': '25',
	# 'cosegLamps': '26',
	# 'cosegVases': '27',
	 'cosegVasesLarge': '28',
	 'cosegChairsLarge': '29',
	 'cosegTeleAliens': '30',
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
			number = 0
			for file in files:
				# load mesh
				mesh = pymesh.load_mesh(os.path.join(root, type, file[:-1] + '.off'))
				label_file = open(os.path.join(root, type, file[:-1] + '_labels.txt'))
				lines = label_file.readlines()
				face_lab = mesh.faces.copy()[:, 1]

				#print(face_lab.shape)
				lab = 1
				for line in range(len(lines)):
					re = line % 2
					if not re:
						st = lines[line + 1]
						label = st.split()
						label_num = list(map(int, label))
						for li in range(len(label_num)):
							face_lab[label_num[li] - 1] = lab
						lab = lab + 1

				# 	if len(lines[line]) > 3:
				#
				# 		if lines[line][-2] == 'a':
				# 			st = lines[line + 1]
				# 			label = st.split()
				# 			label_num = list(map(int, label))
				# 			for li in range(len(label_num)):
				# 				face_lab[label_num[li] - 1] = lab + 1
				#
				# 		if lines[line][-2] == 'b':
				# 			st = lines[line + 1]
				# 			label = st.split()
				# 			label_num = list(map(int, label))
				# 			for li in range(len(label_num)):
				# 				face_lab[label_num[li] - 1] = lab + 2
				#
				# 		if lines[line][-2] == 'c':
				# 			st = lines[line + 1]
				# 			label = st.split()
				# 			label_num = list(map(int, label))
				# 			for li in range(len(label_num)):
				# 				face_lab[label_num[li] - 1] = lab + 3
				#
				# 		if lines[line][-2] == 'd':
				# 			st = lines[line + 1]
				# 			label = st.split()
				# 			label_num = list(map(int, label))
				# 			for li in range(len(label_num)):
				# 				face_lab[label_num[li] - 1] = lab + 4
				#
				# 		if lines[line][-2] == 'e':
				# 			st = lines[line + 1]
				# 			label = st.split()
				# 			label_num = list(map(int, label))
				# 			for li in range(len(label_num)):
				# 				face_lab[label_num[li] - 1] = lab + 5
				#
				# 		if lines[line][-2] == 'f':
				# 			st = lines[line + 1]
				# 			label = st.split()
				# 			label_num = list(map(int, label))
				# 			for li in range(len(label_num)):
				# 				face_lab[label_num[li] - 1] = lab + 6



				# get elements
				vertices = mesh.vertices.copy()
				faces = mesh.faces.copy()
				length = len(face_lab)
				# #move to center
				# center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
				# vertices -= center
				#
				# # normalize
				# max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
				# max_len = np.sqrt(max_len)
				#
				# vertices /= max_len

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
				index = [i for i in range(0, length)]

				# if phrase is 'train':
				# 	random.shuffle(index)

				if npoint >= length:
					if npoint - length > 0:
						for i in range(npoint - length):
							print(i)
							data.append(data[i])
							face_lab.append(face_lab[i])

					data = np.asarray(data)
					face_lab = np.asarray(face_lab)

					number = number + 1
					np.savetxt(data_save + '/' + str(number) + '.txt', data,
					           fmt='%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f')
					np.savetxt(label_save + '/' + str(number) + '.txt', face_lab,
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
						number = number + 1

						np.savetxt(data_save + '/' + str(number) + '.txt', data1,
						           fmt='%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f')
						np.savetxt(label_save + '/' + str(number) + '.txt', face_lab1, fmt="%i")

					fill = index[-npoint:]

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
					number = number + 1
					np.savetxt(data_save + '/' + str(number) + '.txt', data2,
					           fmt='%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f')
					np.savetxt(label_save + '/' + str(number) + '.txt',
					           face_lab2, fmt="%i")

			print('XXX is done!')

			print(phrase + '_' + type + '_' + filename[:-4] + ' is done!')