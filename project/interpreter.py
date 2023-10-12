import cv2
import mediapipe as mp

# 初始化MediaPipe Hand模塊
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 初始化MediaPipe繪圖模塊
mp_drawing = mp.solutions.drawing_utils

# 創建一個VideoCapture對象，指定相機索引，通常0表示默認相機
cap = cv2.VideoCapture(0)

# 檢查攝像頭是否成功打開
if not cap.isOpened():
    print("無法打開攝像頭。請確保相機可用並已連接。")
    exit()

while True:
    # 讀取一個視頻幀
    ret, frame = cap.read()
    
    if not ret:
        print("無法讀取視頻幀。")
        break

    # 將視頻幀轉換為RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用MediaPipe進行手部檢測
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # 在這裡實現手語翻譯的邏輯
            # 可以根據手部地標和連接線的位置來識別特定的手勢
            # 並提供相應的翻譯或操作

            # 繪製手部地標
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # 在視窗中顯示當前視頻幀
    cv2.imshow("Sign Language Interpreter", frame)

    # 檢查是否按下 'q' 鍵，如果是，則退出循環
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
