import cptv
import numpy as np
import cv2
import time

def preprocess(frame):
    frame = np.float32(frame)
    frame[frame > 4000] = 0
    frame = frame - np.median(frame) - 20
    frame = np.uint8(np.clip(frame, 0, 255))
    return frame

#print(cv2.getBuildInformation())
print("Using OpenCL:",cv2.ocl.haveOpenCL())

video = cptv.CPTVReader(open('20171024-103017-akaroa04.cptv', 'rb'))

frames = []

for frame, _ in video:
    frames.append(frame.copy())

first = np.float32(frames[0])
second = np.float32(frames[1])

print(np.mean(first))
print(np.mean(second))

first = preprocess(first)
second = preprocess(second)

opt_flow = cv2.createOptFlow_DualTVL1()

opt_flow.setTau(1 / 4)
opt_flow.setScalesNumber(3)
opt_flow.setWarpingsNumber(3)
opt_flow.setScaleStep(0.5)

height = 120
width = 160

flow = np.zeros([height, width, 2], dtype=np.float32)
cv2.setNumThreads(0)
flow = opt_flow.calc(first, second, flow)

t0 = time.time()
cv2.setNumThreads(0)
for i in range(100):
    flow = opt_flow.calc(first, second, flow)
print("Took: {:.1f}ms".format((time.time()-t0)*1000/100))

print("Flow output std:{:.2f} mean:{:.2f} max:{:.2f} ".format(np.std(flow), np.mean(flow), np.max(flow)))
