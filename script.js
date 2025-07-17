// Chatbot Dinas Arpus Jateng - Main JavaScript
// Version: 2.4.0 (Fixed Sidebar, Gradient Active Menu, Formal Text Menu)
// Refactored for readability and maintainability.

// --- GLOBAL STATE & CONFIGURATION ---
const API_BASE_URL = 'http://localhost:8000'; // Consider changing for production deployment
const MAX_FILES_UPLOAD = 5;
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

let currentUser = { username: "Anonim", role: "public_user", email: "" };
let currentToken = null; // Currently not used for auth, but kept if future auth is added
let selectedChatDocumentId = null;
let currentChatSessionMessages = [];
let selectedUploadFiles = [];
let confirmCallback = null; // Callback for confirmation modal
let allChatDocuments = []; // Store all documents for search functionality

// --- DOM ELEMENTS CACHE ---
// Centralized access to frequently used DOM elements for better performance and readability.
const elements = {
    loadingOverlay: document.getElementById('loading-overlay'),

    navLinks: {
        upload: document.getElementById('nav-upload'),
        documents: document.getElementById('nav-documents'),
        chat: document.getElementById('nav-chat'),
        history: document.getElementById('nav-history'),
        faq: document.getElementById('nav-faq'),
        admin: document.getElementById('nav-admin'),
    },
    navAdminItem: document.getElementById('nav-admin-item'), // The <li> element for admin link

    contentSections: {
        upload: document.getElementById('upload-section'),
        documents: document.getElementById('documents-section'),
        chat: document.getElementById('chat-section'),
        history: document.getElementById('history-section'),
        faq: document.getElementById('faq-section'),
        admin: document.getElementById('admin-section'),
    },

    // Upload Section Elements
    uploadForm: document.getElementById('upload-form'),
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    fileListDisplay: document.getElementById('file-list'),
    uploadBtn: document.getElementById('upload-btn'),

    // Documents Section Elements
    documentsContainer: document.getElementById('documents-container'),

    // Chat Section Elements
    chatDocumentList: document.getElementById('chat-document-list'),
    documentSearchInput: document.getElementById('document-search'),
    predefinedQuestionsContainer: document.getElementById('predefined-questions'),
    predefinedQuestionsList: document.getElementById('questions-list'),
    chatMessagesContainer: document.getElementById('chat-messages'),
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    chatSendBtn: document.getElementById('chat-send-btn'),

    // History Section Elements
    historyContainer: document.getElementById('history-container'),

    // FAQ Section Elements
    faqContainer: document.querySelector('.faq-container'),

    // Admin Section Elements
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

    // Global UI Elements
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

/**
 * Shows or hides the global loading overlay.
 * @param {boolean} show - True to show, false to hide.
 */
function showLoading(show = true) {
    if (elements.loadingOverlay) {
        elements.loadingOverlay.style.display = show ? 'flex' : 'none';
        elements.loadingOverlay.setAttribute('aria-hidden', !show);
        elements.loadingOverlay.setAttribute('aria-busy', show);
    }
}

/**
 * Hides the global loading overlay.
 */
function hideLoading() {
    showLoading(false);
}

/**
 * Displays a temporary alert message to the user.
 * @param {string} message - The message to display.
 * @param {string} type - The type of alert (e.g., 'info', 'success', 'error').
 * @param {number} duration - How long the alert should be visible in milliseconds.
 */
function showAlert(message, type = 'info', duration = 5000) {
    if (!elements.alertContainer) return;

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.textContent = message;

    elements.alertContainer.appendChild(alertDiv);

    // Fade out and remove the alert after duration
    setTimeout(() => {
        alertDiv.style.transform = 'translateX(100%)';
        alertDiv.style.opacity = '0';
        alertDiv.style.transition = 'all 0.3s ease-in';
        setTimeout(() => alertDiv.remove(), 300); // Allow time for transition
    }, duration - 300); // Start fading out 300ms before removal
}

/**
 * Displays a confirmation modal to the user.
 * @param {string} title - The title of the modal.
 * @param {string} message - The message content of the modal.
 * @param {Function} onConfirm - Callback function to execute if user confirms.
 */
function showModal(title, message, onConfirm) {
    const modal = elements.confirmationModal;
    if (!modal || !modal.element) return;

    modal.title.textContent = title;
    modal.message.textContent = message;
    modal.element.style.display = 'flex';
    modal.element.setAttribute('aria-hidden', 'false');
    
    // Focus confirm button after animation starts
    setTimeout(() => {
        modal.confirmBtn.focus();
    }, 100);

    confirmCallback = onConfirm; // Store callback
}

/**
 * Hides the confirmation modal.
 */
function closeModal() {
    const modal = elements.confirmationModal;
    if (!modal || !modal.element) return;

    modal.element.style.opacity = '0';
    modal.element.style.transform = 'scale(0.9)';
    modal.element.setAttribute('aria-hidden', 'true');
    
    // Allow time for CSS transition before setting display to none
    setTimeout(() => {
        modal.element.style.display = 'none';
        modal.element.style.opacity = '';
        modal.element.style.transform = '';
    }, 300);
    confirmCallback = null; // Clear callback
}

/**
 * Formats a date string into a localized, readable format.
 * @param {string} dateString - The date string to format (e.g., ISO 8601).
 * @returns {string} The formatted date string.
 */
function formatDate(dateString) {
    if (!dateString) return 'Tanggal tidak diketahui';
    try {
        return new Date(dateString).toLocaleString('id-ID', {
            day: '2-digit', month: 'short', year: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
    } catch (e) {
        console.error("Failed to format date:", e);
        return dateString; // Return original if formatting fails
    }
}

/**
 * Sets the loading state for a button.
 * @param {HTMLElement} buttonElement - The button element to modify.
 * @param {boolean} isLoading - True to set loading state, false to reset.
 * @param {string} originalText - The original text of the button.
 */
function setButtonLoading(buttonElement, isLoading, originalText = "Submit") {
    if (!buttonElement) return;

    if (isLoading) {
        buttonElement.disabled = true;
        // Store original text in dataset if not already present
        if (!buttonElement.dataset.originalText) {
            buttonElement.dataset.originalText = buttonElement.innerHTML;
        }
        buttonElement.innerHTML = `<span class="loading-spinner-btn" aria-hidden="true"></span> Memproses...`;
        buttonElement.setAttribute('aria-busy', 'true');
    } else {
        buttonElement.disabled = false;
        buttonElement.innerHTML = buttonElement.dataset.originalText || originalText;
        buttonElement.removeAttribute('aria-busy');
        delete buttonElement.dataset.originalText; // Clean up
    }
}

/**
 * Sanitizes text to prevent XSS attacks when inserting into HTML.
 * @param {string} text - The text to sanitize.
 * @returns {string} The sanitized text.
 */
function sanitizeText(text) {
    if (text === null || typeof text === 'undefined') return '';
    const tempDiv = document.createElement('div');
    tempDiv.textContent = String(text); // Use textContent to escape HTML
    return tempDiv.innerHTML;
}

/**
 * Renders an empty state message within a container.
 * @param {HTMLElement} containerElement - The container to render the empty state in.
 * @param {string} message - The message to display.
 */
function renderEmptyState(containerElement, message) {
    if (containerElement) {
        containerElement.innerHTML = `<div class="empty-state"><p>${sanitizeText(message)}</p></div>`;
    }
}

// --- API INTERACTION ---

/**
 * Generic function for making API calls.
 * @param {string} endpoint - The API endpoint (e.g., '/documents').
 * @param {object} options - Fetch API options (method, headers, body).
 * @returns {Promise<any>} The JSON response from the API.
 * @throws {Error} If the API call fails or returns a non-OK status.
 */
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
        // Check for HTTP errors
        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json(); // Try to parse error message from JSON
            } catch (e) {
                // If response is not JSON, use status text
                errorData = { detail: `Server error: ${response.status} ${response.statusText}.` };
            }
            throw new Error(errorData.detail || `HTTP error ${response.status}`);
        }

        // Handle 204 No Content responses
        if (response.status === 204 || response.headers.get("content-length") === "0") {
            return null; // No content to parse
        }

        return await response.json(); // Parse and return JSON response
    } catch (error) {
        console.error('API Call Error:', error.message, `Endpoint: ${endpoint}`);
        throw new Error(error.message || 'Gagal terhubung ke server. Periksa koneksi Anda.');
    }
}

// --- NAVIGATION & SECTION MANAGEMENT ---

/**
 * Shows the main application dashboard layout.
 */
function showDashboard() {
    document.querySelector('.app-main-layout').style.display = 'flex';

    // Check URL for admin demo mode
    const urlParams = new URLSearchParams(window.location.search);
    const isAdminDemo = urlParams.get('is_admin_query') === 'true';

    if (elements.navAdminItem) {
        elements.navAdminItem.style.display = isAdminDemo ? 'list-item' : 'none';
        currentUser.role = isAdminDemo ? 'admin' : 'public_user'; // Update current user role
    }

    navigateToSection('upload'); // Default section to show on load
}

/**
 * Navigates to a specific content section in the UI.
 * @param {string} sectionName - The ID suffix of the section to activate (e.g., 'upload', 'chat').
 */
function navigateToSection(sectionName) {
    // Deactivate all content sections and navigation links
    Object.values(elements.contentSections).forEach(section => {
        if(section) section.classList.remove('active');
    });
    Object.values(elements.navLinks).forEach(link => {
        if (link) {
            link.classList.remove('active');
            link.removeAttribute('aria-current'); // Remove ARIA for non-active links
        }
    });

    // Activate the target section and navigation link
    if (elements.contentSections[sectionName]) {
        elements.contentSections[sectionName].classList.add('active');
    }
    if (elements.navLinks[sectionName]) {
        elements.navLinks[sectionName].classList.add('active');
        elements.navLinks[sectionName].setAttribute('aria-current', 'page'); // Set ARIA for active link
    }

    showLoading(); // Show loading spinner while content loads
    let loadPromise;

    // Load content specific to the navigated section
    switch (sectionName) {
        case 'documents': loadPromise = loadUserDocuments(); break;
        case 'chat': loadPromise = loadDocumentsForChat(); break;
        case 'history': loadPromise = loadUserChatHistory(); break;
        case 'faq': loadPromise = Promise.resolve(); break; // FAQ is static HTML, no API call needed
        case 'admin':
            loadPromise = (currentUser && currentUser.role === 'admin') ? loadAdminDashboardData() : Promise.resolve();
            // If not admin, ensure admin section is cleared or redirects
            if (currentUser && currentUser.role !== 'admin' && elements.contentSections.admin.classList.contains('active')) {
                showAlert("Akses ditolak: Anda bukan administrator.", "error");
                navigateToSection('upload'); // Redirect to a safe section
                loadPromise = Promise.resolve();
            }
            break;
        case 'upload': loadPromise = Promise.resolve(); break; // Upload section is primarily static UI
        default:
            console.warn(`Attempted to navigate to unknown section: ${sectionName}`);
            showAlert(`Bagian '${sectionName}' tidak ditemukan.`, 'error');
            loadPromise = Promise.resolve();
    }

    // Handle loading state after promise resolves or rejects
    if (loadPromise && typeof loadPromise.finally === 'function') {
        loadPromise.catch(err => {
            console.error(`Error loading section ${sectionName}:`, err);
            showAlert(`Gagal memuat bagian ${sectionName}. Coba lagi.`, 'error');
        }).finally(() => {
            hideLoading();
        });
    } else {
        hideLoading(); // Hide if no async load is needed
    }
}

// --- UPLOAD SECTION FUNCTIONS ---

/**
 * Handles file selection from input or drag-and-drop.
 * Filters files by size and type, then updates the UI.
 * @param {FileList} eventFiles - The FileList object from the event.
 */
function handleFileSelection(eventFiles) {
    const currentFileCount = selectedUploadFiles.length;
    const allowedMimeTypes = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // DOCX
        'application/msword', // DOC
        'text/plain'
    ];
    const allowedExtensions = ['pdf', 'docx', 'doc', 'txt'];

    const newFiles = Array.from(eventFiles)
        .filter(file => {
            // Check file size
            if (file.size > MAX_FILE_SIZE_BYTES) {
                showAlert(`File "${sanitizeText(file.name)}" (${(file.size / 1024 / 1024).toFixed(2)}MB) melebihi batas ${MAX_FILE_SIZE_MB}MB.`, 'error', 7000);
                return false;
            }
            // Check file type/extension
            const fileExtension = file.name.split('.').pop().toLowerCase();
            if (!allowedMimeTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
                 showAlert(`Tipe file "${sanitizeText(file.name)}" tidak didukung. Hanya PDF, DOCX, DOC, TXT.`, 'error', 7000);
                 return false;
            }
            // Check if file is already selected (by name and size for simplicity)
            if (selectedUploadFiles.some(existingFile => existingFile.name === file.name && existingFile.size === file.size)) {
                showAlert(`File "${sanitizeText(file.name)}" sudah ada dalam daftar.`, 'info', 5000);
                return false;
            }
            return true;
        })
        .slice(0, MAX_FILES_UPLOAD - currentFileCount); // Limit new files to available slots

    // Add valid new files to selection
    if (newFiles.length > 0) {
        selectedUploadFiles.push(...newFiles);
    }

    // Provide feedback if some files were rejected
    const rejectedCount = Array.from(eventFiles).length - newFiles.length;
    if (rejectedCount > 0 && newFiles.length === 0) { // All new files rejected
        // Message already shown per file
    } else if (selectedUploadFiles.length >= MAX_FILES_UPLOAD && Array.from(eventFiles).length > newFiles.length) {
         showAlert(`Maksimal ${MAX_FILES_UPLOAD} file telah dipilih.`, 'info', 7000);
    }
    updateSelectedFilesUI();
}

/**
 * Updates the UI to display the currently selected files for upload.
 */
function updateSelectedFilesUI() {
    if (!elements.fileListDisplay) return;

    elements.fileListDisplay.innerHTML = ''; // Clear current list

    if (selectedUploadFiles.length === 0) {
        renderEmptyState(elements.fileListDisplay, "Belum ada file dipilih.");
        if (elements.uploadBtn) elements.uploadBtn.disabled = true;
        return;
    }

    // Render each selected file
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
    if (elements.uploadBtn) elements.uploadBtn.disabled = false; // Enable upload button if files are selected
}

/**
 * Removes a file from the selected upload list based on its index.
 * @param {number} indexToRemove - The index of the file to remove.
 */
function removeFileFromSelection(indexToRemove) {
    if (indexToRemove >= 0 && indexToRemove < selectedUploadFiles.length) {
        selectedUploadFiles.splice(indexToRemove, 1);
        updateSelectedFilesUI(); // Re-render UI after removal
        showAlert("File berhasil dihapus dari daftar.", "info", 2000);
    }
}

/**
 * Performs the document upload operation to the backend.
 */
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
        showAlert(`${uploadedCount} dokumen berhasil diunggah!`, 'success', 5000);

        // Clear selected files and update UI
        selectedUploadFiles = [];
        updateSelectedFilesUI();
        if (elements.fileInput) elements.fileInput.value = ''; // Reset file input

        // Suggest chatting with the first uploaded document
        if (uploadedCount > 0 && data.uploaded_documents[0]?.document_id) {
            const firstDoc = data.uploaded_documents[0];
            showModal(
                "Unggah Selesai",
                `Dokumen "${sanitizeText(firstDoc.filename)}" berhasil diunggah. Apakah Anda ingin langsung chat dengan dokumen ini?`,
                () => {
                    selectedChatDocumentId = firstDoc.document_id;
                    navigateToSection('chat'); // Navigate to chat section on confirmation
                }
            );
        }

    } catch (error) {
        showAlert(`Error Unggah: ${error.message}`, 'error', 7000);
    } finally {
        setButtonLoading(elements.uploadBtn, false, "Unggah Dokumen");
    }
}

// --- DOCUMENTS SECTION FUNCTIONS ---

/**
 * Loads and renders the list of all uploaded documents.
 */
async function loadUserDocuments() {
    try {
        const data = await apiCall('/documents');
        renderDocumentList(data.documents || [], elements.documentsContainer, false); // Render in documents section
    } catch (error) {
        showAlert(`Gagal memuat dokumen: ${error.message}`, 'error', 7000);
        renderEmptyState(elements.documentsContainer, 'Gagal memuat dokumen yang tersedia.');
    }
}

/**
 * Renders a list of documents into a specified container.
 * @param {Array<Object>} documents - Array of document objects.
 * @param {HTMLElement} containerElement - The DOM element to render documents into.
 * @param {boolean} isAdminView - True if rendering for admin view (shows delete button).
 */
function renderDocumentList(documents, containerElement, isAdminView = false) {
    if (!containerElement) return;

    containerElement.innerHTML = ''; // Clear existing content

    if (!documents || documents.length === 0) {
        renderEmptyState(containerElement, isAdminView ? 'Tidak ada dokumen di sistem.' : 'Anda belum mengunggah dokumen.');
        return;
    }

    documents.forEach(doc => {
        const docCard = document.createElement('div');
        docCard.className = 'document-card';
        docCard.setAttribute('tabindex', '0'); // Make card focusable
        docCard.setAttribute('aria-labelledby', `doc-title-${doc.id}`);

        const fileSizeMB = doc.file_size ? (doc.file_size / 1024 / 1024).toFixed(2) : 'N/A';

        // Conditional rendering of delete button based on isAdminView
        const deleteButtonHtml = isAdminView ? `
            <button class="btn btn-danger btn-small action-delete" data-doc-id="${doc.id}" data-doc-filename="${sanitizeText(doc.filename)}" aria-label="Hapus dokumen ${sanitizeText(doc.filename)}">Hapus</button>
        ` : '';

        const chatButtonHtml = `
            <button class="btn btn-primary btn-small action-chat" data-doc-id="${doc.id}" data-doc-filename="${sanitizeText(doc.filename)}" aria-label="Mulai chat dengan dokumen ${sanitizeText(doc.filename)}">Chat</button>
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

/**
 * Handles actions (chat or delete) on a document card.
 * @param {string} action - 'chat' or 'delete'.
 * @param {string} documentId - The ID of the document.
 * @param {string} documentFilename - The filename of the document.
 */
async function handleDocumentCardAction(action, documentId, documentFilename) {
    if (action === 'chat') {
        selectedChatDocumentId = documentId;
        const shortFilename = documentFilename.length > 30 ? documentFilename.substring(0, 30) + '...' : documentFilename;
        showAlert(`Dokumen "${sanitizeText(shortFilename)}" dipilih untuk chat`, 'info', 3000);
        navigateToSection('chat');
    } else if (action === 'delete') {
        // Only allow delete if in admin demo mode
        const urlParams = new URLSearchParams(window.location.search);
        const isAdminDemo = urlParams.get('is_admin_query') === 'true';
        if (!isAdminDemo) {
            showAlert("Anda tidak memiliki izin untuk menghapus dokumen.", "error");
            return;
        }

        showModal(
            'Hapus Dokumen',
            `Apakah Anda yakin ingin menghapus dokumen "${sanitizeText(documentFilename)}"? Tindakan ini tidak dapat diurungkan, dan riwayat chat terkait juga akan dihapus.`,
            async () => {
                showLoading();
                try {
                    const endpoint = `/admin/documents/${documentId}?is_admin_query=true`;
                    await apiCall(endpoint, { method: 'DELETE' });
                    showAlert(`Dokumen "${sanitizeText(documentFilename)}" berhasil dihapus.`, 'success');

                    // Reload relevant sections after deletion
                    if (elements.contentSections.documents.classList.contains('active')) {
                        await loadUserDocuments();
                    }
                    if (elements.contentSections.chat.classList.contains('active')) {
                        await loadDocumentsForChat();
                    }
                    if (elements.contentSections.admin.classList.contains('active') && elements.adminTabContents.documents.classList.contains('active')) {
                        await loadAdminAllDocuments();
                    }
                    // Reset chat UI if the deleted document was selected
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

// --- CHAT SECTION FUNCTIONS ---

/**
 * Loads documents for display in the chat sidebar.
 */
async function loadDocumentsForChat() {
    try {
        const data = await apiCall('/documents');
        renderChatDocumentSelectionList(data.documents || []);

        // If a document was previously selected for chat, try to re-select it
        if (selectedChatDocumentId) {
            const stillExists = (data.documents || []).some(doc => doc.id === selectedChatDocumentId);
            if (stillExists) {
                const selectedDocForChat = (data.documents || []).find(doc => doc.id === selectedChatDocumentId);
                activateChatWithDocument(selectedChatDocumentId, selectedDocForChat?.filename);
                // Ensure the UI element for the selected doc gets 'selected' class
                const itemToSelect = elements.chatDocumentList.querySelector(`.chat-document-item[data-doc-id='${selectedChatDocumentId}']`);
                if (itemToSelect) itemToSelect.classList.add('selected');
            } else {
                // If previously selected document no longer exists, reset chat
                selectedChatDocumentId = null;
                resetChatUI();
            }
        } else {
            resetChatUI(); // No document pre-selected, show welcome message
        }
    } catch (error) {
        showAlert(`Gagal memuat daftar dokumen untuk chat: ${error.message}`, 'error', 7000);
        renderEmptyState(elements.chatDocumentList, 'Gagal memuat dokumen.');
        resetChatUI(); // Ensure chat UI is disabled on error
    }
}

/**
 * Renders the list of documents for selection in the chat sidebar.
 * @param {Array<Object>} documents - Array of document objects.
 */
function renderChatDocumentSelectionList(documents) {
    if (!elements.chatDocumentList) return;

    // Only update allChatDocuments if we're rendering the full list (not filtered)
    if (!elements.documentSearchInput || !elements.documentSearchInput.value.trim()) {
        allChatDocuments = documents || []; // Store for search functionality
    }
    
    elements.chatDocumentList.innerHTML = ''; // Clear current list

    if (!documents || documents.length === 0) {
        renderEmptyState(elements.chatDocumentList, 'Belum ada dokumen untuk dichat. Unggah dokumen baru!');
        return;
    }

    documents.forEach(doc => {
        const docItem = document.createElement('div');
        docItem.className = 'chat-document-item';
        docItem.dataset.docId = doc.id;
        docItem.dataset.docFilename = doc.filename;
        docItem.setAttribute('tabindex', '0');
        docItem.setAttribute('role', 'button'); // Indicate it's clickable
        docItem.setAttribute('aria-label', `Pilih dokumen ${sanitizeText(doc.filename)}`);
        docItem.innerHTML = `<div class="document-title">${sanitizeText(doc.filename)}</div>
                             <div class="document-meta">${formatDate(doc.upload_date)}</div>`;
        if (doc.id === selectedChatDocumentId) {
            docItem.classList.add('selected'); // Highlight if currently selected
            docItem.setAttribute('aria-pressed', 'true');
        } else {
            docItem.setAttribute('aria-pressed', 'false');
        }
        elements.chatDocumentList.appendChild(docItem);
    });
}

/**
 * Filters and displays documents based on search query.
 * @param {string} searchQuery - The search query string.
 */
function filterChatDocuments(searchQuery) {
    if (!searchQuery.trim()) {
        // Reset to show all documents when search is cleared
        renderChatDocumentSelectionList(allChatDocuments);
        // Restore selected document state if any
        if (selectedChatDocumentId) {
            const itemToSelect = elements.chatDocumentList.querySelector(`.chat-document-item[data-doc-id='${selectedChatDocumentId}']`);
            if (itemToSelect) {
                itemToSelect.classList.add('selected');
                itemToSelect.setAttribute('aria-pressed', 'true');
            }
        }
        return;
    }

    const filteredDocs = allChatDocuments.filter(doc => 
        doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
    );

    renderChatDocumentSelectionList(filteredDocs);
    // Restore selected document state in filtered results if the selected doc is in the filtered list
    if (selectedChatDocumentId) {
        const itemToSelect = elements.chatDocumentList.querySelector(`.chat-document-item[data-doc-id='${selectedChatDocumentId}']`);
        if (itemToSelect) {
            itemToSelect.classList.add('selected');
            itemToSelect.setAttribute('aria-pressed', 'true');
        }
    }
}

/**
 * Activates the chat interface for a specific document.
 * @param {string} docId - The ID of the document to chat with.
 * @param {string} docFilename - The filename of the document.
 */
function activateChatWithDocument(docId, docFilename) {
    selectedChatDocumentId = docId;
    currentChatSessionMessages = []; // Clear previous chat messages

    // Update UI to highlight selected document
    document.querySelectorAll('#chat-document-list .chat-document-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.docId === docId);
        item.setAttribute('aria-pressed', item.dataset.docId === docId ? 'true' : 'false');
    });

    // Enable and update chat input
    if(elements.chatInput) {
        elements.chatInput.disabled = false;
        elements.chatInput.placeholder = `Bertanya tentang ${sanitizeText(docFilename || "dokumen terpilih")}...`;
        elements.chatInput.focus();
    }
    if(elements.chatSendBtn) elements.chatSendBtn.disabled = false;

    renderChatMessageHistoryUI(); // Show empty chat or existing session messages
    loadPredefinedQuestionsForDocument(docId); // Load predefined questions for the document
}

/**
 * Resets the chat UI to its initial state (no document selected).
 */
function resetChatUI() {
    selectedChatDocumentId = null;
    currentChatSessionMessages = []; // Clear messages

    // Disable and reset chat input
    if(elements.chatInput) {
        elements.chatInput.disabled = true;
        elements.chatInput.value = '';
        elements.chatInput.placeholder = 'Pilih dokumen untuk memulai chat...';
    }
    if(elements.chatSendBtn) elements.chatSendBtn.disabled = true;

    // Hide predefined questions
    if(elements.predefinedQuestionsContainer) elements.predefinedQuestionsContainer.style.display = 'none';

    // Remove selected class from all chat document items
    document.querySelectorAll('#chat-document-list .chat-document-item').forEach(item => {
        item.classList.remove('selected');
        item.setAttribute('aria-pressed', 'false');
    });
    
    // Clear search input and reset document list
    if (elements.documentSearchInput) {
        elements.documentSearchInput.value = '';
    }
    // Reset the document list to show all documents
    if (allChatDocuments.length > 0) {
        renderChatDocumentSelectionList(allChatDocuments);
    }

    renderChatMessageHistoryUI(); // Render the welcome message for chat section
}

/**
 * Loads predefined questions for a specific document.
 * @param {string} docId - The ID of the document.
 */
async function loadPredefinedQuestionsForDocument(docId) {
    if(!elements.predefinedQuestionsContainer || !elements.predefinedQuestionsList) return;

    elements.predefinedQuestionsContainer.style.display = 'none'; // Hide until questions are loaded
    elements.predefinedQuestionsList.innerHTML = ''; // Clear previous questions

    try {
        const response = await apiCall(`/predefined-questions/${docId}`);
        if (response && response.questions && response.questions.length > 0) {
            response.questions.forEach(questionText => {
                const qItem = document.createElement('button');
                qItem.className = 'question-item';
                qItem.textContent = questionText;
                qItem.setAttribute('role', 'button');
                qItem.setAttribute('aria-label', `Ajukan pertanyaan: ${sanitizeText(questionText)}`);
                elements.predefinedQuestionsList.appendChild(qItem);
            });
            elements.predefinedQuestionsContainer.style.display = 'block'; // Show if questions are available
        }
    } catch (error) {
        console.warn('Gagal memuat pertanyaan umum:', error.message);
        // Do not show a full alert for this as it's a secondary feature
    }
}

/**
 * Renders the chat message history in the UI.
 */
function renderChatMessageHistoryUI() {
    if (!elements.chatMessagesContainer) return;

    elements.chatMessagesContainer.innerHTML = ''; // Clear current messages

    if (currentChatSessionMessages.length === 0) {
        // Display a welcome message if no chat history
        const welcomeMessage = selectedChatDocumentId
            ? `Mulai bertanya tentang dokumen yang dipilih!`
            : 'Pilih dokumen dari daftar di samping untuk memulai chat.';
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'chat-welcome';
        welcomeDiv.innerHTML = `<p>${sanitizeText(welcomeMessage)}</p>`;
        elements.chatMessagesContainer.appendChild(welcomeDiv);
        return;
    }

    // Add each message to the UI
    currentChatSessionMessages.forEach(msg => addMessageToChatUI(msg.content, msg.sender, msg.timestamp, false));
    // Scroll to the latest message
    elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
}

/**
 * Adds a single message to the chat UI.
 * @param {string} content - The message text.
 * @param {'user'|'assistant'} sender - Who sent the message.
 * @param {string} timestamp - ISO string timestamp of the message.
 * @param {boolean} scrollToBottom - Whether to scroll the chat window to the bottom.
 */
function addMessageToChatUI(content, sender, timestamp, scrollToBottom = true) {
    if (!elements.chatMessagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.setAttribute('role', 'status'); // For accessibility

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content; // Use textContent for safety

    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = new Date(timestamp).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(metaDiv);
    elements.chatMessagesContainer.appendChild(messageDiv);

    // Keep track of messages in the session
    currentChatSessionMessages.push({ content, sender, timestamp });

    if (scrollToBottom) {
        elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
    }
}

/**
 * Submits a chat message to the backend AI.
 * @param {string} messageContent - The content of the message.
 * @param {boolean} isPredefined - True if the message comes from a predefined question.
 */
async function submitChatMessage(messageContent, isPredefined = false) {
    if (!selectedChatDocumentId) {
        showAlert("Pilih dokumen terlebih dahulu untuk memulai chat.", "error");
        return;
    }
    if (!messageContent.trim()) {
        showAlert("Ketik pertanyaan Anda pada kolom input.", "info");
        return;
    }

    // Add user's message to UI immediately
    addMessageToChatUI(messageContent, 'user', new Date().toISOString());
    if (elements.chatInput && !isPredefined) elements.chatInput.value = ''; // Clear input if not predefined

    setButtonLoading(elements.chatSendBtn, true, "Kirim");
    if (elements.chatInput) elements.chatInput.disabled = true; // Disable input while AI is thinking

    try {
        const responseData = await apiCall('/chat', {
            method: 'POST',
            body: JSON.stringify({
                message: messageContent,
                document_ids: [selectedChatDocumentId], // Send selected doc ID
                is_predefined: isPredefined
            }),
        });
        addMessageToChatUI(responseData.response, 'assistant', new Date().toISOString());
    } catch (error) {
        showAlert(`Error Chat: ${error.message}`, 'error', 7000);
        addMessageToChatUI('Maaf, terjadi kesalahan internal saat memproses pertanyaan Anda. Silakan coba lagi.', 'assistant', new Date().toISOString());
    } finally {
        setButtonLoading(elements.chatSendBtn, false, "Kirim");
        if (elements.chatInput) {
            elements.chatInput.disabled = false;
            elements.chatInput.focus(); // Re-focus input for next question
        }
    }
}

// --- HISTORY SECTION FUNCTIONS ---

/**
 * Loads and renders the user's chat history.
 */
async function loadUserChatHistory() {
    try {
        const data = await apiCall('/history');
        renderChatHistoryList(data.history || []);
    } catch (error) {
        showAlert(`Gagal memuat riwayat chat: ${error.message}`, 'error', 7000);
        renderEmptyState(elements.historyContainer, 'Gagal memuat riwayat percakapan Anda.');
    }
}

/**
 * Renders the list of chat history items.
 * @param {Array<Object>} historyItems - Array of chat history objects.
 */
function renderChatHistoryList(historyItems) {
    if (!elements.historyContainer) return;

    elements.historyContainer.innerHTML = ''; // Clear existing content

    // Add "Clear All" button if there are history items
    if (historyItems && historyItems.length > 0) {
        const clearAllBtn = document.createElement('button');
        clearAllBtn.className = 'btn btn-danger btn-small';
        clearAllBtn.style.marginBottom = '1.5rem';
        clearAllBtn.textContent = 'Hapus Semua Riwayat';
        clearAllBtn.setAttribute('aria-label', 'Hapus semua riwayat chat');
        clearAllBtn.addEventListener('click', () => {
            showModal(
                'Hapus Semua Riwayat',
                'Apakah Anda yakin ingin menghapus semua riwayat chat? Tindakan ini tidak dapat diurungkan.',
                async () => {
                    await deleteAllChatHistory();
                }
            );
        });
        elements.historyContainer.appendChild(clearAllBtn);
    }

    if (!historyItems || historyItems.length === 0) {
        renderEmptyState(elements.historyContainer, 'Belum ada riwayat chat yang tersimpan.');
        return;
    }

    historyItems.forEach(item => {
        const historyItemDiv = document.createElement('div');
        historyItemDiv.className = 'history-item';
        historyItemDiv.style.position = 'relative'; // For positioning the delete button

        let docContextInfo = "Konteks Umum";
        // Attempt to parse document_ids if available
        if (item.document_ids && typeof item.document_ids === 'string') {
            try {
                const docIds = JSON.parse(item.document_ids);
                if (docIds && docIds.length > 0) {
                    docContextInfo = `Dok. ID: ${docIds.map(sanitizeText).join(', ')}`;
                }
            } catch (e) {
                console.warn("Failed to parse document_ids from history item:", item.document_ids, e);
            }
        }

        const userDisplay = item.username ? `<span>Pengguna: ${sanitizeText(item.username)}</span>` : '<span>Pengguna: Anonim</span>';

        // Add delete button for individual items
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-danger btn-small';
        deleteBtn.style.position = 'absolute';
        deleteBtn.style.top = '1rem';
        deleteBtn.style.right = '1rem';
        deleteBtn.style.fontSize = '0.8rem';
        deleteBtn.style.padding = '0.4rem 0.8rem';
        deleteBtn.textContent = 'Hapus';
        deleteBtn.setAttribute('aria-label', `Hapus riwayat chat ini (ID: ${item.id})`);
        deleteBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation(); // Prevent card click event if any
            showModal(
                'Hapus Riwayat',
                'Apakah Anda yakin ingin menghapus riwayat chat ini?',
                async () => {
                    await deleteChatHistory(item.id);
                }
            );
        });

        historyItemDiv.innerHTML = `
            <div class="history-question"><strong>Q:</strong> <span class="question-text">${sanitizeText(item.message)}</span></div>
            <div class="history-answer"><span class="answer-text">${sanitizeText(item.response)}</span></div>
            <div class="history-meta">
                ${userDisplay}
                <span>${sanitizeText(docContextInfo)}</span>
                <span>${formatDate(item.timestamp)}</span>
            </div>
        `;

        historyItemDiv.appendChild(deleteBtn);
        elements.historyContainer.appendChild(historyItemDiv);
    });
}

/**
 * Deletes a single chat history item.
 * @param {number} historyId - The ID of the history item to delete.
 */
async function deleteChatHistory(historyId) {
    showLoading();
    try {
        await apiCall(`/history/${historyId}`, { method: 'DELETE' });
        showAlert('Riwayat chat berhasil dihapus', 'success', 4000);
        await loadUserChatHistory(); // Reload the history list
    } catch (error) {
        showAlert(`Gagal menghapus riwayat: ${error.message}`, 'error', 6000);
    } finally {
        hideLoading();
    }
}

/**
 * Deletes all chat history items.
 */
async function deleteAllChatHistory() {
    showLoading();
    try {
        const response = await apiCall('/history', { method: 'DELETE' });
        const deletedCount = response.deleted_count || 0;
        showAlert(`Semua riwayat chat berhasil dihapus (${deletedCount} item)`, 'success', 5000);
        await loadUserChatHistory(); // Reload the history list (should now be empty)
    } catch (error) {
        showAlert(`Gagal menghapus semua riwayat: ${error.message}`, 'error', 6000);
    } finally {
        hideLoading();
    }
}

// --- ADMIN SECTION FUNCTIONS ---

/**
 * Loads and displays administrative dashboard data (stats, recent activity).
 */
async function loadAdminDashboardData() {
    // Check if user is actually an admin (redundant, but good for defensive programming)
    const urlParams = new URLSearchParams(window.location.search);
    const isAdminDemo = urlParams.get('is_admin_query') === 'true';

    if (!isAdminDemo) {
        showAlert("Akses ditolak: Anda bukan administrator.", "error");
        navigateToSection('upload'); // Redirect if not admin
        return;
    }

    try {
        const statsData = await apiCall('/admin/stats?is_admin_query=true');
        if (elements.adminStats.documents) elements.adminStats.documents.textContent = statsData.total_documents !== undefined ? statsData.total_documents : '0';
        if (elements.adminStats.chats) elements.adminStats.chats.textContent = statsData.total_chats !== undefined ? statsData.total_chats : '0';

        // Determine which tab is active and load its content
        const activeTabBtn = elements.adminTabsContainer?.querySelector('.admin-tab-btn.active');
        const activeTabName = activeTabBtn ? activeTabBtn.dataset.tab : 'documents'; // Default to 'documents'
        await switchAdminTab(activeTabName, true); // Force load content for the active tab

    } catch (error) {
        showAlert(`Gagal memuat statistik admin: ${error.message}`, 'error', 7000);
        // Display error state in stats cards
        if (elements.adminStats.documents) elements.adminStats.documents.textContent = 'Error';
        if (elements.adminStats.chats) elements.adminStats.chats.textContent = 'Error';
    }
}

/**
 * Switches between admin tabs (Documents, Activity) and loads their content.
 * @param {string} tabName - The name of the tab to switch to ('documents' or 'activity').
 * @param {boolean} forceLoad - True to force data reload for the tab.
 */
async function switchAdminTab(tabName, forceLoad = false) {
    // Hide all tab contents and deactivate all tab buttons
    document.querySelectorAll('.admin-tab-content').forEach(content => content.classList.remove('active'));
    document.querySelectorAll('.admin-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
    });

    // Show the selected tab content and activate its button
    if (elements.adminTabContents[tabName]) {
        elements.adminTabContents[tabName].classList.add('active');
    }
    const tabButton = elements.adminTabsContainer?.querySelector(`.admin-tab-btn[data-tab='${tabName}']`);
    if (tabButton) {
        tabButton.classList.add('active');
        tabButton.setAttribute('aria-selected', 'true');
    }

    // Load content for the selected tab if forced or not already loaded
    if (forceLoad) {
        showLoading();
        let loadPromise;
        if (tabName === 'documents') {
            loadPromise = loadAdminAllDocuments();
        } else if (tabName === 'activity') {
            loadPromise = loadAdminSystemActivity();
        } else {
            loadPromise = Promise.resolve();
        }

        // Handle completion of loading
        if (loadPromise && typeof loadPromise.finally === 'function') {
            loadPromise.catch(err => console.error(`Error loading admin tab ${tabName}:`, err))
                       .finally(() => hideLoading());
        } else {
            hideLoading();
        }
    }
}

/**
 * Loads and renders all documents in the system for the admin view.
 */
async function loadAdminAllDocuments() {
    try {
        const data = await apiCall('/admin/documents?is_admin_query=true');
        renderDocumentList(data.documents || [], elements.adminDocumentsList, true); // Use true for isAdminView
    } catch (error) {
        showAlert(`Gagal memuat daftar semua dokumen (admin): ${error.message}`, 'error', 7000);
        renderEmptyState(elements.adminDocumentsList, 'Gagal memuat daftar dokumen.');
    }
}

/**
 * Loads and renders recent system activity (chat history and uploads) for admin view.
 */
async function loadAdminSystemActivity() {
    try {
        const data = await apiCall('/admin/stats?is_admin_query=true'); // Re-use admin stats endpoint for activity
        renderAdminActivityList(data.recent_activity || []);
    } catch (error) {
        showAlert(`Gagal memuat aktivitas sistem: ${error.message}`, 'error', 7000);
        renderEmptyState(elements.adminActivityList, 'Gagal memuat aktivitas sistem.');
    }
}

/**
 * Renders the list of admin activities.
 * @param {Array<Object>} activities - Array of activity objects.
 */
function renderAdminActivityList(activities) {
    if(!elements.adminActivityList) return;

    elements.adminActivityList.innerHTML = ''; // Clear existing content

    if (!activities || activities.length === 0) {
        renderEmptyState(elements.adminActivityList, 'Tidak ada aktivitas terbaru.');
        return;
    }

    // Display recent activities (e.g., top 30)
    const recentActivities = activities.slice(0, 30);
    recentActivities.forEach(item => {
        const actItem = document.createElement('div');
        actItem.className = 'admin-item';

        // Construct document context info
        let docContextInfo = "Konteks Umum";
        if (item.type === 'chat' && item.document_ids && typeof item.document_ids === 'string') {
            try {
                const docIds = JSON.parse(item.document_ids);
                if (docIds && docIds.length > 0) {
                    docContextInfo = `Dok. ID: ${docIds.map(sanitizeText).join(', ')}`;
                }
            } catch (e) { /* ignore parse error for malformed data */ }
        } else if (item.type === 'upload' && item.filename) {
             docContextInfo = `File: ${sanitizeText(item.filename)}`;
        }


        actItem.innerHTML = `
            <div class="admin-item-info">
                <strong>${item.type === 'chat' ? 'Chat' : 'Unggahan'} (${sanitizeText(docContextInfo)})</strong>
                <span class="item-filename">
                    ${item.type === 'chat' ? `Pertanyaan: ${sanitizeText(item.description)}` : `Nama File: ${sanitizeText(item.description)}`}
                </span>
                <small class="item-meta">${formatDate(item.timestamp)} oleh ${sanitizeText(item.username)}</small>
            </div>`;
        elements.adminActivityList.appendChild(actItem);
    });
}

// --- EVENT LISTENERS & APP INITIALIZATION ---

/**
 * Initializes all global event listeners for the application.
 */
function initializeEventListeners() {
    // Main Navigation: Handle clicks on sidebar links
    Object.entries(elements.navLinks).forEach(([sectionName, navElement]) => {
        if (navElement) {
            navElement.addEventListener('click', (e) => {
                e.preventDefault(); // Prevent default link behavior
                if (!e.currentTarget.classList.contains('active')) { // Only navigate if not already active
                    navigateToSection(sectionName);
                }
            });
        }
    });

    // Upload Section: File input and drag-and-drop
    const uploadArea = elements.uploadArea;
    if (uploadArea) {
        uploadArea.addEventListener('click', () => elements.fileInput?.click());
        uploadArea.addEventListener('keypress', (e) => {
            // Allow keyboard activation (Enter/Space)
            if (e.key === 'Enter' || e.key === ' ') {
                elements.fileInput?.click();
            }
        });
        // Drag-and-drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault(); // Prevent default browser drag behavior
                e.stopPropagation(); // Stop event propagation
            }, false); // Use capture phase for these events
        });
        uploadArea.addEventListener('dragenter', () => uploadArea.classList.add('dragover'));
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            uploadArea.classList.remove('dragover');
            handleFileSelection(e.dataTransfer.files); // Process dropped files
        });
    }
    elements.fileInput?.addEventListener('change', (e) => handleFileSelection(e.target.files));
    elements.uploadForm?.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent default form submission
        await performUploadDocuments();
    });
    elements.fileListDisplay?.addEventListener('click', (e) => {
        const removeButton = e.target.closest('.file-remove');
        if (removeButton) {
            const index = parseInt(removeButton.dataset.index, 10);
            if (!isNaN(index)) removeFileFromSelection(index);
        }
    });

    // Document & Chat Actions: Handle buttons within document cards and chat sidebar
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
            submitChatMessage(targetQuestion.textContent, true); // Mark as predefined
        }
    });
    elements.chatForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = elements.chatInput.value.trim();
        if (message) await submitChatMessage(message);
    });
    
    // Document Search: Filter documents as user types
    elements.documentSearchInput?.addEventListener('input', (e) => {
        filterChatDocuments(e.target.value);
    });

    // FAQ Section: Accordion functionality
    elements.faqContainer?.addEventListener('click', (e) => {
        const questionButton = e.target.closest('.faq-question');
        if (questionButton) {
            const isActive = questionButton.classList.toggle('active');
            questionButton.setAttribute('aria-expanded', isActive);
            const answer = questionButton.nextElementSibling; // The .faq-answer div
            if (answer) {
                // Set max-height to scrollHeight if active, else to 0 for collapse transition
                answer.style.maxHeight = isActive ? answer.scrollHeight + "px" : null;
            }
        }
    });

    // Admin Section: Tab switching and document deletion
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

    // Confirmation Modal: Button actions
    elements.confirmationModal.confirmBtn?.addEventListener('click', () => {
        if (typeof confirmCallback === 'function') confirmCallback(); // Execute stored callback
        closeModal();
    });
    elements.confirmationModal.cancelBtn?.addEventListener('click', closeModal);
    // Close modal if background is clicked
    elements.confirmationModal.element?.addEventListener('click', (e) => {
        if (e.target === elements.confirmationModal.element) closeModal();
    });
    // Trap focus within modal (basic implementation)
    elements.confirmationModal.element?.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
            return;
        }
        if (e.key === 'Tab') {
            const focusableElements = elements.confirmationModal.element.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
            const firstFocusableEl = focusableElements[0];
            const lastFocusableEl = focusableElements[focusableElements.length - 1];

            if (e.shiftKey) { // Shift + Tab
                if (document.activeElement === firstFocusableEl) {
                    lastFocusableEl.focus();
                    e.preventDefault();
                }
            } else { // Tab
                if (document.activeElement === lastFocusableEl) {
                    firstFocusableEl.focus();
                    e.preventDefault();
                }
            }
        }
    });
}

/**
 * Initializes the entire application when the DOM is ready.
 */
async function initializeApp() {
    console.log("Menginisialisasi aplikasi Chatbot Dinas Arpus Jateng...");
    showLoading(); // Show loading overlay during initialization
    try {
        initializeEventListeners(); // Set up all event handlers
        showDashboard(); // Render the initial dashboard view
    } catch (error) {
        console.error("Terjadi kesalahan selama inisialisasi aplikasi:", error);
        showAlert("Terjadi kesalahan saat memulai aplikasi. Silakan coba lagi.", "error");
    } finally {
        hideLoading(); // Hide loading overlay regardless of success or failure
    }
}

// Attach the initialization function to the DOMContentLoaded event
document.addEventListener('DOMContentLoaded', initializeApp);