let defects = [];
let logs = [];
let lastStatus = null;

// SocketIO for alerts
const socket = io();

// Status elements
const statusIndicator = document.getElementById('status-indicator');
const alertIndicator = document.getElementById('alert-indicator');
const liveStatus = document.getElementById('live-status');
const videoFeed = document.getElementById('video-feed');
const defectGallery = document.getElementById('defect-gallery');
const logsList = document.getElementById('logs-list');

socket.on('connect', () => console.log('Socket connected'));

socket.on('detection_update', (data) => {
    updateStatus(data.status, data.is_defect);
    if (data.is_defect && data.defect_image) {
        addDefectImage(data.defect_image);
        playAlertSound();
    }
});

socket.on('error', (data) => {
    console.error('Server error:', data.message);
    alert('Connection error: ' + data.message);
});

// Update video feed
function updateVideoFeed(base64) {
// videoFeed.src = 'data:image/jpeg;base64,' + base64 + '?t=' + Date.now();
}

// Update status indicators
function updateStatus(status, isDefect) {
    const isCorrect = !isDefect;
    
    // Main status
    statusIndicator.textContent = status;
    statusIndicator.className = 'status ' + (isCorrect ? 'correct' : 'defect');
    
    // Live badge
    liveStatus.textContent = isCorrect ? '✅ Correct' : '❌ ' + status;
    liveStatus.className = 'live-badge ' + (isCorrect ? 'correct' : 'defect');
    
    // Alert indicator
    if (isDefect) {
        alertIndicator.classList.remove('hidden');
    } else {
        alertIndicator.classList.add('hidden');
    }
}

// Add defect to gallery
function addDefectImage(imagePath) {
    if (!defects.includes(imagePath)) {
        defects.push(imagePath);
        const img = document.createElement('img');
        img.src = '/defect_images/' + encodeURIComponent(imagePath.split('/').pop());
        img.title = imagePath;
        img.onclick = () => window.open(img.src, '_blank');
        defectGallery.insertBefore(img, defectGallery.firstChild);
    }
}

// Load logs
function loadLogs() {
    fetch('/logs')
        .then(res => res.json())
        .then(data => {
            logsList.innerHTML = '';
            data.logs.forEach(log => {
                const entry = document.createElement('div');
                entry.className = 'log-entry ' + (log.type !== 'None' ? 'defect' : '');
                entry.innerHTML = `
                    <strong>${new Date(log.timestamp).toLocaleString()}</strong><br>
                    ${log.type} - <a href="/defect_images/${encodeURIComponent(log.image.split('/').pop())}" target="_blank">View Image</a>
                `;
                logsList.appendChild(entry);
            });
        });
}

// Event listeners
document.getElementById('refresh-logs').onclick = loadLogs;
document.getElementById('clear-defects').onclick = () => {
    defects = [];
    defectGallery.innerHTML = '';
};

// Poll predictions for real-time status/sound
let lastStatus = null;
function pollStatus() {
    fetch('/api/predict')
        .then(res => res.json())
        .then(data => {
            updateStatus(data.status, data.is_defect);
            if (data.is_defect && data.status != lastStatus) {
                playAlertSound();
            }
            lastStatus = data.status;
        })
        .catch(e => console.error('Poll error:', e));
}

// Play alert sound
function playAlertSound() {
    const audio = document.getElementById('alertSound');
    audio.play().catch(e => console.log('Audio play failed:', e));
}

setInterval(pollStatus, 500);  // 2 FPS poll

// Initial load
loadLogs();
setInterval(loadLogs, 10000);

// Handle page visibility
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        loadLogs();
    }
});

