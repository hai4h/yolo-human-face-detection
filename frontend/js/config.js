/**
 * config.js
 * Configuration settings for the application
 */

// Check if the device is mobile
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Export application settings
const config = {
    // Detection settings
    showPersons: true,
    showFaces: true,
    showConfidence: true,
    showFaceNames: true, // Hiển thị tên khuôn mặt tương đồng / Show similar face names
    personColor: '#e74c3c',
    faceColor: '#2ecc71',
    
    // Server settings
    serverUrl: '/process_frame',
    
    // Performance settings
    frameRate: isMobile ? 20 : 30, // Lower FPS on mobile devices
    
    // Device info
    isMobile: isMobile,
    
    // Nhãn kết quả nhận diện / Detection label settings
    desktopLabelFontSize: 10,     // Kích thước chữ trên desktop (px)
    mobileLabelFontSize: 20,      // Kích thước chữ trên thiết bị di động (px)
    labelPadding: 4,              // Padding cho nhãn (px)
    labelMargin: 6,               // Khoảng cách từ nhãn đến khung (px)
    borderWidth: 2                // Độ dày viền khung (px)
};

export default config;