document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Sidebar Toggle Logic ---
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    if (sidebar && sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('-translate-x-full');
        });
    }

    // --- 2. Replace Chat Naming Function ---
    if (typeof generateChatName === 'function') {
        window.originalGenerateChatName = generateChatName;
        window.generateChatName = function() {
            const backendMap = {
                'llama.cpp': 'CPP',
                'ollama': 'OL',
                'safetensors': 'ST',
                'api': 'API'
            };
            const backendShort = backendMap[window.currentBackend] || 'LLM';
            const modelSelect = document.getElementById('model-select');
            const modelName = modelSelect && modelSelect.value 
                ? modelSelect.options[modelSelect.selectedIndex].text.replace('.gguf', '') 
                : 'Unknown Model';
            return `${backendShort} - ${modelName}`;
        }
    }

    // --- 3. Replace Stream Handling Functions ---
    // Functions removed to allow index.html to handle streaming
});
