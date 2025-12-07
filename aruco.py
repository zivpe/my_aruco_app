import streamlit as st
import cv2
import numpy as np
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray

# --- הגדרות עמוד ---
st.set_page_config(page_title="ArUco Area Calculator", layout="wide")
st.title("🌱ArUco מדידת שטח אובייקט באמצעות ")
st.write("העלה תמונה הכוללת סמן (בגודל 5x5 ס\"מ) ואת האובייקט אותו תרצה למדוד.")

# --- פונקציות עזר ---

@st.cache_data
def segment_image_kmeans(img, k=3, attempts=10):
    """
    מבצעת סגמנטציה של תמונה באמצעות אלגוריתם K-Means.
    מבוסס על הדוגמה במחברת
    """
    # המרת התמונה למערך דו-ממדי של פיקסלים (MxN, 3)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # הגדרת קריטריונים לעצירה
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # ביצוע K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # המרה חזרה ל-uint8
    centers = np.uint8(centers)
    # שיטוח מערך התוויות
    labels = labels.flatten()

    # יצירת התמונה המסווגת (צביעת כל פיקסל בצבע המרכז שלו)
    segmented_image = centers[labels]
    # החזרת התמונה לממדים המקוריים
    segmented_image = segmented_image.reshape(img.shape)

    return segmented_image, labels, centers

# --- סרגל צד (Sidebar) ---
st.sidebar.header("הגדרות סגמנטציה")
# בחירת מספר הקלאסטרים (K) - כמו במחברת
k_value = st.sidebar.slider('מספר צבעים (K):', min_value=2, max_value=6, value=3, help="לכמה צבעים דומיננטיים לחלק את התמונה?")
attempts_value = st.sidebar.slider('מספר ניסיונות K-Means:', min_value=1, max_value=10, value=5)

# --- חלק ראשי: העלאת קובץ ---
uploaded_file = st.file_uploader("בחר תמונה...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # קריאת התמונה מהקובץ שהועלה
    # שימוש ב-opencv לקריאת ה-bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # המרה ל-RGB עבור הצגה ב-Streamlit ועבור עיבוד ב-scikit-image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # הצגת התמונה המקורית
    st.subheader("תמונה מקורית")
    st.image(image_rgb, use_column_width=True)

    # --- שלב 1: זיהוי ArUco וחישוב יחס ---
    st.header("שלב 1: זיהוי סמן ")
    
    # הגדרת המילון והגלאי (בהתאם לקוד המקורי שלך)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # המרה לגווני אפור לצורך זיהוי ה-ArUco
    grayscale_img = img_as_ubyte(rgb2gray(image_rgb))
    
    # זיהוי הסמנים
    corners, ids, rejected = detector.detectMarkers(grayscale_img)

    cm2_per_px2_ratio = None

    if corners:
        # ציור המסגרת סביב הסמן שזוהה
        img_with_aruco = image_rgb.copy()
        int_corners = np.int32(corners)
        cv2.polylines(img_with_aruco, int_corners, True, (0, 255, 0), 5)
        st.image(img_with_aruco, caption="סמן ArUco שזוהה", use_column_width=True)

        # חישוב שטח הסמן בפיקסלים
        aruco_area_px = cv2.contourArea(corners[0])
        
        # שטח ידוע במציאות: 5 ס"מ * 5 ס"מ = 25 סמ"ר
        aruco_area_cm_real = 25
        
        # חישוב היחס: כמה סמ"ר שווה כל פיקסל בודד
        cm2_per_px2_ratio = aruco_area_cm_real / aruco_area_px
        
        st.success(f" סמן זוהה בהצלחה! שטח בפיקסלים : {aruco_area_px:.1f}. יחס המרה: {cm2_per_px2_ratio:.6f} סמ\"ר לפיקסל.")

    else:
        st.error("לא נמצא סמן ArUco בתמונה. וודא שהסמן ברור ומסוג 5X5.")
        st.stop() # עצור את הריצה אם אין סמן

    # --- שלב 2: סגמנטציה (K-Means) ובחירת אובייקט ---
    if cm2_per_px2_ratio is not None:
        st.header("שלב 2: בידוד האובייקט ומדידה")
        st.write("האלגוריתם מחלק את התמונה למספר צבעים עיקריים. עליך לבחור איזה צבע מייצג את האובייקט שלך.")

        # ביצוע הסגמנטציה כאשר המשתמש לוחץ על כפתור (כדי לא להריץ מחדש בכל שינוי סליידר)
        if st.button('בצע סגמנטציה (K-Means)'):
            with st.spinner('מבצע סגמנטציה...'):
                segmented_img_rgb, labels, centers = segment_image_kmeans(image_rgb, k=k_value, attempts=attempts_value)
            
            st.subheader("תמונה מסווגת (Segmented Image)")
            st.image(segmented_img_rgb, use_column_width=True)
            
            st.divider()
            st.subheader("בחירת האובייקט למדידה")
            
            # --- יצירת ממשק בחירה למשתמש ---
            # נציג למשתמש את ה"צבעים" (centers) שהאלגוריתם מצא, והוא יבחר איזה מהם הוא העלה
            
            clusters_data = []
            cols = st.columns(k_value) # יצירת עמודות להצגת דוגמיות הצבע

            for i in range(k_value):
                # יצירת ריבוע צבע קטן להמחשה
                color_swatch = np.zeros((50, 50, 3), dtype=np.uint8)
                color_swatch[:, :] = centers[i]
                
                # ספירת כמה פיקסלים שייכים לקלאסטר הזה
                count = np.sum(labels == i)
                clusters_data.append({"id": i, "color": centers[i], "count": count})

                with cols[i]:
                    # הצגת דוגמית הצבע והמזהה שלה
                    st.image(color_swatch, caption=f"Cluster {i}")
                    st.caption(f"פיקסלים: {count}")

            # תיבת בחירה למשתמש
            selected_cluster_id = st.selectbox(
                "בחר את מספר הקלאסטר (Cluster ID) שמייצג את האובייקט (למשל, העלה הירוק):",
                options=[c["id"] for c in clusters_data]
            )

            # --- שלב 3: חישוב התוצאה הסופית ---
            
            # חישוב מספר הפיקסלים של הקלאסטר הנבחר
            object_pixel_count = clusters_data[selected_cluster_id]["count"]
            
            # המרה לסנטימטרים רבועים באמצעות היחס שמצאנו קודם
            real_area_cm2 = object_pixel_count * cm2_per_px2_ratio

            st.divider()
            # הצגת התוצאה בגדול
            st.metric(label="שטח האובייקט הנבחר במציאות", value=f"{real_area_cm2:.2f} סמ\"ר")
            
            # (אופציונלי) הצגת האובייקט הנבחר בלבד לבקרה
            mask = (labels == selected_cluster_id).reshape(image_rgb.shape[:2])
            final_object_viz = np.zeros_like(image_rgb)
            final_object_viz[mask] = image_rgb[mask]
            st.subheader("בקרה: האובייקט שנבחר למדידה")
            st.image(final_object_viz, use_column_width=True)
