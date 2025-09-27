import React, { useState } from 'react';

interface ImageGeneratorProps {
    prompt: string;
    onGenerate: (prompt: string, model: string, settings: any) => void;
    onClose: () => void;
}

const ImageGenerator: React.FC<ImageGeneratorProps> = ({ prompt, onGenerate, onClose }) => {
    const [currentPrompt, setCurrentPrompt] = useState(prompt);
    const [selectedModel, setSelectedModel] = useState('stable-diffusion-xl');
    const [width, setWidth] = useState(512);
    const [height, setHeight] = useState(512);
    const [steps, setSteps] = useState(20);
    const [cfgScale, setCfgScale] = useState(7.5);

    const models = [
        'stable-diffusion-xl',
        'stable-diffusion-1.5',
        'midjourney-style',
        'realistic-vision',
        'anime-style'
    ];

    const handleGenerate = () => {
        onGenerate(currentPrompt, selectedModel, {
            width,
            height,
            steps,
            cfgScale
        });
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-2xl">
                <h3 className="text-2xl font-bold mb-4 text-white">Generate Image</h3>
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Prompt:</label>
                        <textarea 
                            value={currentPrompt}
                            onChange={(e) => setCurrentPrompt(e.target.value)}
                            rows={3}
                            className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Model:</label>
                        <select 
                            value={selectedModel} 
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        >
                            {models.map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Width:</label>
                            <input 
                                type="number" 
                                value={width}
                                onChange={(e) => setWidth(parseInt(e.target.value))}
                                min={64}
                                max={1024}
                                className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Height:</label>
                            <input 
                                type="number" 
                                value={height}
                                onChange={(e) => setHeight(parseInt(e.target.value))}
                                min={64}
                                max={1024}
                                className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            />
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Steps:</label>
                            <input 
                                type="number" 
                                value={steps}
                                onChange={(e) => setSteps(parseInt(e.target.value))}
                                min={1}
                                max={100}
                                className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">CFG Scale:</label>
                            <input 
                                type="number" 
                                value={cfgScale}
                                onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                                min={1}
                                max={20}
                                step={0.5}
                                className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            />
                        </div>
                    </div>
                </div>
                <div className="flex justify-end space-x-4 mt-6">
                    <button 
                        onClick={onClose} 
                        className="px-4 py-2 rounded bg-gray-600 text-white hover:bg-gray-500 transition-colors"
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={handleGenerate} 
                        className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-500 transition-colors"
                    >
                        Generate Image
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ImageGenerator;
