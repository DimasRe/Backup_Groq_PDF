// Chatbot Dinas Arpus Jateng - Main JavaScript
// Version: 2.4.0 (Fixed Sidebar, Gradient Active Menu, Formal Text Menu)

// --- GLOBAL STATE & CONFIGURATION ---
const API_BASE_URL = 'http://localhost:8000';
const MAX_FILES_UPLOAD = 5;
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

let currentUser = { username: "Anonim", role: "public_user", email: "" };
let currentToken = null;
let selectedChatDocumentId = null;
let currentChatSessionMessages = [];
let selectedUploadFiles = [];
let confirmCallback = null;

// --- DOM ELEMENTS CACHE ---
const elements = {
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // appMainLayout: document.querySelector('.app-main-layout'), // New: Main layout container
    // Removed dashboardSection, as app-main-layout now serves as primary layout.
    // Dashboard section can still be used to wrap inner content within main-content-area

    navLinks: {
        upload: document.getElementById('nav-upload'),
        documents: document.getElementById('nav-documents'),
        chat: document.getElementById('nav-chat'),
        history: document.getElementById('nav-history'),
        faq: document.getElementById('nav-faq'),
        admin: document.getElementById('nav-admin'),
    },
    navAdminItem: document.getElementById('nav-admin-item'),
    contentSections: {
        upload: document.getElementById('upload-section'),
        documents: document.getElementById('documents-section'),
        chat: document.getElementById('chat-section'),
        history: document.getElementById('history-section'),
        faq: document.getElementById('faq-section'),
        admin: document.getElementById('admin-section'),
    },
    uploadForm: document.getElementById('upload-form'),
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    fileListDisplay: document.getElementById('file-list'),
    uploadBtn: document.getElementById('upload-btn'),
    documentsContainer: document.getElementById('documents-container'),
    chatDocumentList: document.getElementById('chat-document-list'),
    predefinedQuestionsContainer: document.getElementById('predefined-questions'),
    predefinedQuestionsList: document.getElementById('questions-list'),
    chatMessagesContainer: document.getElementById('chat-messages'),
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    historyContainer: document.getElementById('history-container'),
    faqContainer: document.querySelector('.faq-container'),
    adminStats: {
        documents: document.getElementById('stat-documents'),
        chats: document.getElementById('stat-chats'),
    },
    adminTabsContainer: document.querySelector('.admin-tabs'),
    adminTabContents: {
        documents: document.getElementById('admin-documents-tab'),
        activity: document.getElementById('admin-activity-tab'),
    },
    adminDocumentsList: document.getElementById('admin-documents-list'),
    adminActivityList: document.getElementById('admin-activity-list'),
    alertContainer: document.getElementById('alert-container'),
    confirmationModal: {
        element: document.getElementById('confirmation-modal'),
        title: document.getElementById('modal-title'),
        message: document.getElementById('modal-message'),
        cancelBtn: document.getElementById('modal-cancel'),
        confirmBtn: document.getElementById('modal-confirm'),
    }
};

// --- UTILITY FUNCTIONS ---
function showLoading(show = true) {
    if (elements.loadingOverlay) {
        elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    }
}

function hideLoading() {
    if (elements.loadingOverlay) {
        elements.loadingOverlay.style.display = 'none';
    }
}

function showAlert(message, type = 'info', duration = 5000) {
    if (!elements.alertContainer) return;
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.textContent = message;
    elements.alertContainer.appendChild(alertDiv);
    setTimeout(() => {
        alertDiv.style.opacity = '0';
        setTimeout(() => alertDiv.remove(), 400);
    }, duration - 400);
}

function showModal(title, message, onConfirm) {
    const modal = elements.confirmationModal;
    if (!modal || !modal.element) return;
    modal.title.textContent = title;
    modal.message.textContent = message;
    modal.element.classList.add('active');
    modal.element.style.display = 'flex';
    confirmCallback = onConfirm;
}

function closeModal() {
    const modal = elements.confirmationModal;
    if (!modal || !modal.element) return;
    modal.element.classList.remove('active');
    setTimeout(() => {
        modal.element.style.display = 'none';
    }, 300);
    confirmCallback = null;
}

function formatDate(dateString) {
    if (!dateString) return 'Tanggal tidak diketahui';
    try {
        return new Date(dateString).toLocaleString('id-ID', {
            day: '2-digit', month: 'short', year: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
    } catch (e) {
        return dateString;
    }
}

function setButtonLoading(buttonElement, isLoading, originalText = "Submit") {
    if (!buttonElement) return;
    if (isLoading) {
        buttonElement.disabled = true;
        if (!buttonElement.dataset.originalText) {
            buttonElement.dataset.originalText = buttonElement.innerHTML;
        }
        buttonElement.innerHTML = `<span class="loading-spinner-btn"></span> Memproses...`;
    } else {
        buttonElement.disabled = false;
        buttonElement.innerHTML = buttonElement.dataset.originalText || originalText;
    }
}

function sanitizeText(text) {
    if (text === null || typeof text === 'undefined') return '';
    const tempDiv = document.createElement('div');
    tempDiv.textContent = String(text);
    return tempDiv.innerHTML;
}

function renderEmptyState(containerElement, message) {
    if (containerElement) {
        containerElement.innerHTML = `<div class="empty-state"><p>${sanitizeText(message)}</p></div>`;
    }
}

// --- API FUNCTIONS ---
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
        ...options,
        headers: {
            ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
            ...options.headers,
        },
    };

    try {
        const response = await fetch(url, config);
        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch (e) {
                errorData = { detail: `Server error: ${response.status} ${response.statusText}.` };
            }
            throw new Error(errorData.detail || `HTTP error ${response.status}`);
        }
        if (response.status === 204 || response.headers.get("content-length") === "0") {
            return null;
        }
        return await response.json();
    } catch (error) {
        console.error('API Call Error:', error.message, `Endpoint: ${endpoint}`);
        throw new Error(error.message || 'Gagal terhubung ke server. Periksa koneksi Anda.');
    }
}

function showDashboard() {
    // elements.dashboardSection.style.display = 'flex'; // This is no longer the main layout control
    document.querySelector('.app-main-layout').style.display = 'flex'; // Show the main layout

    const urlParams = new URLSearchParams(window.location.search);
    const isAdminDemo = urlParams.get('is_admin_query') === 'true';
    if (elements.navAdminItem) {
        elements.navAdminItem.style.display = isAdminDemo ? 'list-item' : 'none';
        currentUser.role = isAdminDemo ? 'admin' : 'public_user';
    }

    navigateToSection('upload');
}

function navigateToSection(sectionName) {
    Object.values(elements.contentSections).forEach(section => {
        if(section) section.classList.remove('active');
    });
    Object.values(elements.navLinks).forEach(link => {
        if (link) {
            link.classList.remove('active');
            link.removeAttribute('aria-current');
        }
    });

    if (elements.contentSections[sectionName]) {
        elements.contentSections[sectionName].classList.add('active');
    }
    if (elements.navLinks[sectionName]) {
        elements.navLinks[sectionName].classList.add('active');
        elements.navLinks[sectionName].setAttribute('aria-current', 'page');
    }

    showLoading();
    let loadPromise;
    switch (sectionName) {
        case 'documents': loadPromise = loadUserDocuments(); break;
        case 'chat': loadPromise = loadDocumentsForChat(); break;
        case 'history': loadPromise = loadUserChatHistory(); break;
        case 'faq': loadPromise = Promise.resolve(); break;
        case 'admin': loadPromise = (currentUser && currentUser.role === 'admin') ? loadAdminDashboardData() : Promise.resolve(); break;
        case 'upload': loadPromise = Promise.resolve(); break;
        default:
            console.warn(`Navigasi ke bagian tidak dikenal: ${sectionName}`);
            loadPromise = Promise.resolve();
    }
    if (loadPromise && typeof loadPromise.finally === 'function') {
        loadPromise.catch(err => {
            console.error(`Error loading section ${sectionName}:`, err);
            showAlert(`Gagal memuat bagian ${sectionName}. Coba lagi.`, 'error');
        }).finally(() => {
            hideLoading();
        });
    } else {
        hideLoading();
    }
}

function handleFileSelection(eventFiles) {
    const currentFileCount = selectedUploadFiles.length;
    const newFiles = Array.from(eventFiles)
        .filter(file => {
            if (file.size > MAX_FILE_SIZE_BYTES) {
                showAlert(`File "${sanitizeText(file.name)}" (${(file.size / 1024 / 1024).toFixed(2)}MB) melebihi batas ${MAX_FILE_SIZE_MB}MB.`, 'error', 7000);
                return false;
            }
            const allowedMimeTypes = [
                'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/msword',
                'text/plain'
            ];
            const fileExtension = file.name.split('.').pop().toLowerCase();
            const allowedExtensions = ['pdf', 'docx', 'doc', 'txt'];

            if (!allowedMimeTypes.includes(file.type) && !allowedExtensions.includes(fileExtension) ) {
                 showAlert(`Tipe file "${sanitizeText(file.name)}" tidak didukung. Hanya PDF, DOCX, DOC, TXT.`, 'error', 7000);
                 return false;
            }
            return true;
        })
        .slice(0, MAX_FILES_UPLOAD - currentFileCount);

    if (newFiles.length > 0) {
        selectedUploadFiles.push(...newFiles);
    }

    const rejectedCount = Array.from(eventFiles).length - newFiles.length;
    if (rejectedCount > 0 && (newFiles.length + currentFileCount) < MAX_FILES_UPLOAD) {
        showAlert(`Beberapa file tidak ditambahkan karena melebihi batas ukuran atau tipe tidak didukung.`, 'info', 7000);
    } else if (selectedUploadFiles.length >= MAX_FILES_UPLOAD && Array.from(eventFiles).length > newFiles.length) {
         showAlert(`Maksimal ${MAX_FILES_UPLOAD} file telah dipilih.`, 'info', 7000);
    }
    updateSelectedFilesUI();
    if (elements.uploadBtn) elements.uploadBtn.disabled = selectedUploadFiles.length === 0;
}

function updateSelectedFilesUI() {
    if (!elements.fileListDisplay) return;
    elements.fileListDisplay.innerHTML = '';

    if (selectedUploadFiles.length === 0) {
        renderEmptyState(elements.fileListDisplay, "Belum ada file dipilih.");
        if (elements.uploadBtn) elements.uploadBtn.disabled = true;
        return;
    }

    selectedUploadFiles.forEach((file, index) => {
        const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <span class="file-icon" aria-hidden="true">📄</span>
                <span class="file-name">${sanitizeText(file.name)}</span>
                <span class="file-size">(${fileSizeMB} MB)</span>
            </div>
            <button type="button" class="file-remove" data-index="${index}" aria-label="Hapus file ${sanitizeText(file.name)}">×</button>
        `;
        elements.fileListDisplay.appendChild(fileItem);
    });
    if (elements.uploadBtn) elements.uploadBtn.disabled = false;
}

function removeFileFromSelection(indexToRemove) {
    selectedUploadFiles.splice(indexToRemove, 1);
    updateSelectedFilesUI();
}

async function performUploadDocuments() {
    if (selectedUploadFiles.length === 0) {
        showAlert("Tidak ada file yang dipilih untuk diupload.", "info");
        return;
    }
    setButtonLoading(elements.uploadBtn, true, "Unggah Dokumen");

    const formData = new FormData();
    selectedUploadFiles.forEach(file => formData.append('files', file));

    try {
        const data = await apiCall('/upload', { method: 'POST', body: formData });
        const uploadedCount = data.uploaded_documents ? data.uploaded_documents.length : 0;
        showAlert(`${uploadedCount} dokumen berhasil diunggah! ${sanitizeText(data.message || '')}`, 'success');

        selectedUploadFiles = [];
        updateSelectedFilesUI();
        if (elements.fileInput) elements.fileInput.value = '';
        if (elements.uploadBtn) elements.uploadBtn.disabled = true;

        if (uploadedCount > 0 && data.uploaded_documents[0]?.document_id) {
            const firstDoc = data.uploaded_documents[0];
            showModal(
                "Unggah Selesai",
                `Dokumen "${sanitizeText(firstDoc.filename)}" berhasil diunggah. Apakah Anda ingin langsung chat dengan dokumen ini?`,
                () => {
                    selectedChatDocumentId = firstDoc.document_id;
                    navigateToSection('chat');
                }
            );
        }

    } catch (error) {
        showAlert(`Error Unggah: ${error.message}`, 'error');
    } finally {
        setButtonLoading(elements.uploadBtn, false, "Unggah Dokumen");
    }
}

async function loadUserDocuments() {
    try {
        const data = await apiCall('/documents');
        renderDocumentList(data.documents || [], elements.documentsContainer, false);
    } catch (error) {
        showAlert(`Gagal memuat dokumen: ${error.message}`, 'error');
        renderEmptyState(elements.documentsContainer, 'Gagal memuat dokumen yang tersedia.');
    } finally {
    }
}

function renderDocumentList(documents, containerElement, isUserSpecificView = false) {
    if (!containerElement) return;
    containerElement.innerHTML = '';

    if (!documents || documents.length === 0) {
        renderEmptyState(containerElement, isUserSpecificView ? 'Anda belum mengunggah dokumen.' : 'Tidak ada dokumen di sistem.');
        return;
    }

    documents.forEach(doc => {
        const docCard = document.createElement('div');
        docCard.className = 'document-card';
        docCard.setAttribute('tabindex', '0');
        docCard.setAttribute('aria-labelledby', `doc-title-${doc.id}`);

        const fileSizeMB = doc.file_size ? (doc.file_size / 1024 / 1024).toFixed(2) : 'N/A';

        const urlParams = new URLSearchParams(window.location.search);
        const isAdminDemo = urlParams.get('is_admin_query') === 'true';
        const deleteButtonHtml = isAdminDemo ? `
            <button class="btn btn-danger btn-small action-delete" data-doc-id="${doc.id}" data-doc-filename="${sanitizeText(doc.filename)}">Hapus</button>
        ` : '';

        const chatButtonHtml = `
            <button class="btn btn-primary btn-small action-chat" data-doc-id="${doc.id}" data-doc-filename="${sanitizeText(doc.filename)}">Chat</button>
        `;

        docCard.innerHTML = `
            <div class="document-header">
                <h3 class="document-title" id="doc-title-${doc.id}">${sanitizeText(doc.filename)}</h3>
                <p class="document-meta">
                    Diupload: ${formatDate(doc.upload_date)}<br>
                    Ukuran: ${fileSizeMB} MB
                </p>
            </div>
            <div class="document-actions">${chatButtonHtml} ${deleteButtonHtml}</div>
        `;
        containerElement.appendChild(docCard);
    });
}

async function handleDocumentCardAction(action, documentId, documentFilename) {
    if (action === 'chat') {
        selectedChatDocumentId = documentId;
        showAlert(`Dokumen "${sanitizeText(documentFilename)}" dipilih. Pindah ke tab Chat.`, 'info', 3000);
        navigateToSection('chat');
    } else if (action === 'delete') {
        const urlParams = new URLSearchParams(window.location.search);
        const isAdminDemo = urlParams.get('is_admin_query') === 'true';
        if (!isAdminDemo) {
            showAlert("Anda tidak memiliki izin untuk menghapus dokumen.", "error");
            return;
        }
        showModal(
            'Hapus Dokumen',
            `Apakah Anda yakin ingin menghapus dokumen "${sanitizeText(documentFilename)}"? Tindakan ini tidak dapat diurungkan.`,
            async () => {
                showLoading();
                try {
                    const endpoint = `/admin/documents/${documentId}?is_admin_query=true`;
                    await apiCall(endpoint, { method: 'DELETE' });
                    showAlert(`Dokumen "${sanitizeText(documentFilename)}" berhasil dihapus.`, 'success');

                    if (elements.contentSections.documents.classList.contains('active')) loadUserDocuments();
                    if (elements.contentSections.chat.classList.contains('active')) loadDocumentsForChat();
                    if (elements.contentSections.admin.classList.contains('active') && elements.adminTabContents.documents.classList.contains('active')) {
                        loadAdminAllDocuments();
                    }
                    if (selectedChatDocumentId === documentId) {
                        resetChatUI();
                    }
                } catch (error) {
                    showAlert(`Gagal menghapus dokumen: ${error.message}`, 'error');
                } finally {
                    hideLoading();
                }
            }
        );
    }
}

async function loadDocumentsForChat() {
    try {
        const data = await apiCall('/documents');
        renderChatDocumentSelectionList(data.documents || []);
        if (selectedChatDocumentId) {
            const stillExists = (data.documents || []).some(doc => doc.id === selectedChatDocumentId);
            if (stillExists) {
                const selectedDocForChat = (data.documents || []).find(doc => doc.id === selectedChatDocumentId);
                activateChatWithDocument(selectedChatDocumentId, selectedDocForChat?.filename);
                const itemToSelect = elements.chatDocumentList.querySelector(`.chat-document-item[data-doc-id='${selectedChatDocumentId}']`);
                if (itemToSelect) itemToSelect.classList.add('selected');
            } else {
                selectedChatDocumentId = null;
                resetChatUI();
            }
        } else {
            resetChatUI();
        }
    } catch (error) {
        showAlert(`Gagal memuat daftar dokumen untuk chat: ${error.message}`, 'error');
        renderEmptyState(elements.chatDocumentList, 'Gagal memuat dokumen.');
        resetChatUI();
    } finally {
    }
}

function renderChatDocumentSelectionList(documents) {
    elements.chatDocumentList.innerHTML = '';
    if (!documents || documents.length === 0) {
        renderEmptyState(elements.chatDocumentList, 'Belum ada dokumen untuk dichat.');
        return;
    }
    documents.forEach(doc => {
        const docItem = document.createElement('div');
        docItem.className = 'chat-document-item';
        docItem.dataset.docId = doc.id;
        docItem.dataset.docFilename = doc.filename;
        docItem.setAttribute('tabindex', '0');
        docItem.setAttribute('role', 'button');
        docItem.innerHTML = `<div class="document-title">${sanitizeText(doc.filename)}</div>
                             <div class="document-meta">${formatDate(doc.upload_date)}</div>`;
        if (doc.id === selectedChatDocumentId) {
            docItem.classList.add('selected');
        }
        elements.chatDocumentList.appendChild(docItem);
    });
}

function activateChatWithDocument(docId, docFilename) {
    selectedChatDocumentId = docId;
    currentChatSessionMessages = [];

    document.querySelectorAll('#chat-document-list .chat-document-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.docId === docId);
        item.setAttribute('aria-pressed', item.dataset.docId === docId ? 'true' : 'false');
    });
    if(elements.chatInput) {
        elements.chatInput.disabled = false;
        elements.chatInput.placeholder = `Bertanya tentang ${sanitizeText(docFilename || "dokumen terpilih")}...`;
        elements.chatInput.focus();
    }
    if(elements.chatSendBtn) elements.chatSendBtn.disabled = false;


    renderChatMessageHistoryUI();
    loadPredefinedQuestionsForDocument(docId);
}

function resetChatUI() {
    selectedChatDocumentId = null;
    currentChatSessionMessages = [];
    if(elements.chatInput) {
        elements.chatInput.disabled = true;
        elements.chatInput.value = '';
        elements.chatInput.placeholder = 'Pilih dokumen untuk memulai chat...';
    }
    if(elements.chatSendBtn) elements.chatSendBtn.disabled = true;
    if(elements.predefinedQuestionsContainer) elements.predefinedQuestionsContainer.style.display = 'none';
    renderChatMessageHistoryUI();
}

async function loadPredefinedQuestionsForDocument(docId) {
    if(!elements.predefinedQuestionsContainer || !elements.predefinedQuestionsList) return;
    elements.predefinedQuestionsContainer.style.display = 'none';
    elements.predefinedQuestionsList.innerHTML = '';
    try {
        const response = await apiCall(`/predefined-questions/${docId}`);
        if (response && response.questions && response.questions.length > 0) {
            response.questions.forEach(questionText => {
                const qItem = document.createElement('button');
                qItem.className = 'question-item';
                qItem.textContent = questionText;
                elements.predefinedQuestionsList.appendChild(qItem);
            });
            elements.predefinedQuestionsContainer.style.display = 'block';
        }
    } catch (error) {
        console.warn('Gagal memuat pertanyaan umum:', error.message);
    }
}

function renderChatMessageHistoryUI() {
    if (!elements.chatMessagesContainer) return;
    elements.chatMessagesContainer.innerHTML = '';
    if (currentChatSessionMessages.length === 0) {
        const welcomeMessage = selectedChatDocumentId
            ? `Mulai bertanya tentang dokumen yang dipilih!`
            : 'Pilih dokumen dari daftar di samping untuk memulai chat.';
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'chat-welcome';
        welcomeDiv.innerHTML = `<p>${sanitizeText(welcomeMessage)}</p>`;
        elements.chatMessagesContainer.appendChild(welcomeDiv);
        return;
    }
    currentChatSessionMessages.forEach(msg => addMessageToChatUI(msg.content, msg.sender, msg.timestamp, false));
    elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
}

function addMessageToChatUI(content, sender, timestamp, scrollToBottom = true) {
    if (!elements.chatMessagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = new Date(timestamp).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(metaDiv);
    elements.chatMessagesContainer.appendChild(messageDiv);

    if (scrollToBottom) {
        elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
    }
    currentChatSessionMessages.push({ content, sender, timestamp });
}


async function submitChatMessage(messageContent, isPredefined = false) {
    if (!selectedChatDocumentId) {
        showAlert("Pilih dokumen terlebih dahulu untuk memulai chat.", "error");
        return;
    }
    if (!messageContent.trim()) {
        showAlert("Ketik pertanyaan Anda pada kolom input.", "info");
        return;
    }

    addMessageToChatUI(messageContent, 'user', new Date().toISOString());
    if (elements.chatInput && !isPredefined) elements.chatInput.value = '';
    setButtonLoading(elements.chatSendBtn, true, "Kirim");
    if (elements.chatInput) elements.chatInput.disabled = true;

    try {
        const responseData = await apiCall('/chat', {
            method: 'POST',
            body: JSON.stringify({
                message: messageContent,
                document_ids: [selectedChatDocumentId],
                is_predefined: isPredefined
            }),
        });
        addMessageToChatUI(responseData.response, 'assistant', new Date().toISOString());
    } catch (error) {
        showAlert(`Error Chat: ${error.message}`, 'error');
        addMessageToChatUI('Maaf, terjadi kesalahan internal saat memproses pertanyaan Anda. Silakan coba lagi.', 'assistant', new Date().toISOString());
    } finally {
        setButtonLoading(elements.chatSendBtn, false, "Kirim");
        if (elements.chatInput) {
            elements.chatInput.disabled = false;
            elements.chatInput.focus();
        }
    }
}

async function loadUserChatHistory() {
    try {
        const data = await apiCall('/history');
        renderChatHistoryList(data.history || []);
    } catch (error) {
        showAlert(`Gagal memuat riwayat chat: ${error.message}`, 'error');
        renderEmptyState(elements.historyContainer, 'Gagal memuat riwayat percakapan Anda.');
    } finally {
    }
}

function renderChatHistoryList(historyItems) {
    if (!elements.historyContainer) return;
    elements.historyContainer.innerHTML = '';
    if (!historyItems || historyItems.length === 0) {
        renderEmptyState(elements.historyContainer, 'Belum ada riwayat chat yang tersimpan.');
        return;
    }
    historyItems.forEach(item => {
        const historyItemDiv = document.createElement('div');
        historyItemDiv.className = 'history-item';

        let docContextInfo = "Konteks Umum";
        if (item.document_ids && typeof item.document_ids === 'string') {
            try {
                const docIds = JSON.parse(item.document_ids);
                if (docIds && docIds.length > 0) {
                    docContextInfo = `Dok. ID: ${docIds.map(sanitizeText).join(', ')}`;
                }
            } catch (e) { console.warn("Gagal parse document_ids dari riwayat:", item.document_ids); }
        }

        const userDisplay = item.username ? `<span>Pengguna: ${sanitizeText(item.username)}</span>` : '<span>Pengguna: Anonim</span>';

        historyItemDiv.innerHTML = `
            <div class="history-question"><strong>Q:</strong> <span class="question-text">${sanitizeText(item.message)}</span></div>
            <div class="history-answer"><span class="answer-text">${sanitizeText(item.response)}</span></div>
            <div class="history-meta">
                ${userDisplay}
                <span>${sanitizeText(docContextInfo)}</span>
                <span>${formatDate(item.timestamp)}</span>
            </div>
        `;
        elements.historyContainer.appendChild(historyItemDiv);
    });
}

async function loadAdminDashboardData() {
    const urlParams = new URLSearchParams(window.location.search);
    const isAdminDemo = urlParams.get('is_admin_query') === 'true';

    if (!isAdminDemo) {
        showAlert("Akses ditolak: Anda bukan administrator.", "error");
        navigateToSection('upload');
        return;
    }

    try {
        const statsData = await apiCall('/admin/stats?is_admin_query=true');
        if (elements.adminStats.documents) elements.adminStats.documents.textContent = statsData.total_documents !== undefined ? statsData.total_documents : 'N/A';
        if (elements.adminStats.chats) elements.adminStats.chats.textContent = statsData.total_chats !== undefined ? statsData.total_chats : 'N/A';

        const activeTabBtn = elements.adminTabsContainer?.querySelector('.admin-tab-btn.active');
        const activeTabName = activeTabBtn ? activeTabBtn.dataset.tab : 'documents';
        await switchAdminTab(activeTabName, true);

    } catch (error) {
        showAlert(`Gagal memuat statistik admin: ${error.message}`, 'error');
        if (elements.adminStats.documents) elements.adminStats.documents.textContent = 'Error';
        if (elements.adminStats.chats) elements.adminStats.chats.textContent = 'Error';
    } finally {
    }
}

async function switchAdminTab(tabName, forceLoad = false) {
    document.querySelectorAll('.admin-tab-content').forEach(content => content.classList.remove('active'));
    document.querySelectorAll('.admin-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
    });

    if (elements.adminTabContents[tabName]) {
        elements.adminTabContents[tabName].classList.add('active');
    }
    const tabButton = elements.adminTabsContainer?.querySelector(`.admin-tab-btn[data-tab='${tabName}']`);
    if (tabButton) {
        tabButton.classList.add('active');
        tabButton.setAttribute('aria-selected', 'true');
    }

    if (forceLoad) {
        showLoading();
        let loadPromise;
        if (tabName === 'documents') loadPromise = loadAdminAllDocuments();
        else if (tabName === 'activity') loadPromise = loadAdminSystemActivity();
        else loadPromise = Promise.resolve(hideLoading());

        if (loadPromise && typeof loadPromise.finally === 'function') {
            loadPromise.catch(err => console.error(`Error loading admin tab ${tabName}:`, err))
                       .finally(() => hideLoading());
        }
    }
}

async function loadAdminAllDocuments() {
    try {
        const data = await apiCall('/admin/documents?is_admin_query=true');
        renderDocumentList(data.documents || [], elements.adminDocumentsList, false);
    } catch (error) {
        showAlert(`Gagal memuat daftar semua dokumen (admin): ${error.message}`, 'error');
        renderEmptyState(elements.adminDocumentsList, 'Gagal memuat daftar dokumen.');
    }
}

async function loadAdminSystemActivity() {
    try {
        const data = await apiCall('/history?is_admin_query=true');
        renderAdminActivityList(data.history || []);
    } catch (error) {
        showAlert(`Gagal memuat aktivitas sistem: ${error.message}`, 'error');
        renderEmptyState(elements.adminActivityList, 'Gagal memuat aktivitas sistem.');
    }
}

function renderAdminActivityList(activities) {
    if(!elements.adminActivityList) return;
    elements.adminActivityList.innerHTML = '';
    if (!activities || activities.length === 0) {
        renderEmptyState(elements.adminActivityList, 'Tidak ada aktivitas terbaru.');
        return;
    }
    const recentActivities = activities.slice(0, 30);
    recentActivities.forEach(item => {
        const actItem = document.createElement('div');
        actItem.className = 'admin-item';
        const questionPreview = item.message.substring(0, 70) + (item.message.length > 70 ? '...' : '');
        const activityUser = item.username || 'Anonim';

        let docContextInfo = "Konteks Umum";
        if (item.document_ids && typeof item.document_ids === 'string') {
            try {
                const docIds = JSON.parse(item.document_ids);
                if (docIds && docIds.length > 0) {
                    docContextInfo = `Dok. ID: ${docIds.map(sanitizeText).join(', ')}`;
                }
            } catch (e) { /* ignore parse error */ }
        }

        actItem.innerHTML = `
            <div class="admin-item-info">
                <strong>Chat (${sanitizeText(docContextInfo)})</strong>
                <span class="item-filename">Pertanyaan: ${sanitizeText(questionPreview)}</span>
                <small class="item-meta">${formatDate(item.timestamp)} oleh ${sanitizeText(activityUser)}</small>
            </div>`;
        elements.adminActivityList.appendChild(actItem);
    });
}
// --- EVENT LISTENERS & APP INITIALIZATION ---

function initializeEventListeners() {
    // Main Navigation
    Object.entries(elements.navLinks).forEach(([sectionName, navElement]) => {
        if (navElement) {
            navElement.addEventListener('click', (e) => {
                e.preventDefault();
                if (!e.currentTarget.classList.contains('active')) {
                    navigateToSection(sectionName);
                }
            });
        }
    });

    // Upload
    const uploadArea = elements.uploadArea;
    if (uploadArea) {
        uploadArea.addEventListener('click', () => elements.fileInput?.click());
        uploadArea.addEventListener('keypress', (e) => { if (e.key === 'Enter' || e.key === ' ') elements.fileInput?.click(); });
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => { e.preventDefault(); e.stopPropagation(); }, false);
        });
        uploadArea.addEventListener('dragenter', () => uploadArea.classList.add('dragover'));
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            uploadArea.classList.remove('dragover');
            handleFileSelection(e.dataTransfer.files);
        });
    }
    elements.fileInput?.addEventListener('change', (e) => handleFileSelection(e.target.files));
    elements.uploadForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        await performUploadDocuments();
    });
    elements.fileListDisplay?.addEventListener('click', (e) => {
        const removeButton = e.target.closest('.file-remove');
        if (removeButton) {
            const index = parseInt(removeButton.dataset.index, 10);
            if (!isNaN(index)) removeFileFromSelection(index);
        }
    });

    // Document & Chat Actions
    elements.documentsContainer?.addEventListener('click', (e) => {
        const targetButton = e.target.closest('button.action-chat, button.action-delete');
        if (targetButton) {
            const action = targetButton.classList.contains('action-chat') ? 'chat' : 'delete';
            handleDocumentCardAction(action, targetButton.dataset.docId, targetButton.dataset.docFilename);
        }
    });
    elements.chatDocumentList?.addEventListener('click', (e) => {
        const targetItem = e.target.closest('.chat-document-item');
        if (targetItem?.dataset.docId) {
            activateChatWithDocument(targetItem.dataset.docId, targetItem.dataset.docFilename);
        }
    });
    elements.predefinedQuestionsList?.addEventListener('click', (e) => {
        const targetQuestion = e.target.closest('.question-item');
        if (targetQuestion) {
            submitChatMessage(targetQuestion.textContent, true);
        }
    });
    elements.chatForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = elements.chatInput.value.trim();
        if (message) await submitChatMessage(message);
    });

    // FAQ Accordion
    elements.faqContainer?.addEventListener('click', (e) => {
        const questionButton = e.target.closest('.faq-question');
        if (questionButton) {
            const isActive = questionButton.classList.toggle('active');
            const answer = questionButton.nextElementSibling;
            if (answer) {
                answer.style.maxHeight = isActive ? answer.scrollHeight + "px" : null;
            }
        }
    });

    // Admin
    elements.adminTabsContainer?.addEventListener('click', (e) => {
        const targetButton = e.target.closest('.admin-tab-btn');
        if (targetButton && !targetButton.classList.contains('active')) {
            switchAdminTab(targetButton.dataset.tab, true);
        }
    });
    elements.adminDocumentsList?.addEventListener('click', (e) => {
        const targetButton = e.target.closest('button.action-delete');
        if (targetButton) {
            handleDocumentCardAction('delete', targetButton.dataset.docId, targetButton.dataset.docFilename);
        }
    });


    // Modal
    elements.confirmationModal.confirmBtn?.addEventListener('click', () => {
        if (typeof confirmCallback === 'function') confirmCallback();
        closeModal();
    });
    elements.confirmationModal.cancelBtn?.addEventListener('click', closeModal);
    elements.confirmationModal.element?.addEventListener('click', (e) => {
        if (e.target === elements.confirmationModal.element) closeModal();
    });
}

async function initializeApp() {
    console.log("Menginisialisasi aplikasi Chatbot Dinas Arpus Jateng...");
    showLoading();
    try {
        initializeEventListeners();
        showDashboard();
    } catch (error) {
        console.error("Terjadi kesalahan selama inisialisasi aplikasi:", error);
        showAlert("Terjadi kesalahan saat memulai aplikasi. Silakan coba lagi.", "error");
    } finally {
        hideLoading();
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);