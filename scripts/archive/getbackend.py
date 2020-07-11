import cv2

cap = cv2.VideoCapture(r'E:\Peptide Logic\Writhing\BW_MWT_191107_M4_R2.mp4')
# for backend in cv2.videoio_registry.getBackends():
#     print(cv2.videoio_registry.getBackendName(backend))
print(cap.getBackendName())