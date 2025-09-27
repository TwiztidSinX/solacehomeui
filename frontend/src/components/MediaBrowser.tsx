import React, { useState, useEffect, useCallback } from 'react';

interface MediaItem {
    Id: string;
    Name: string;
    Type: 'Movie' | 'Series' | 'Season' | 'Episode' | 'BoxSet';
    Year?: number;
    Overview?: string;
    ImageUrl?: string;
    IsFolder: boolean;
}

interface MediaBrowserProps {
    query: string;
    onPlay: (mediaId: string, mediaType: string) => void;
    onClose: () => void;
    settings: {
        mediaServerUrl?: string;
        mediaServerApiKey?: string;
    };
}

const MediaBrowser: React.FC<MediaBrowserProps> = ({ query, onPlay, onClose, settings }) => {
    const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
    const [currentPath, setCurrentPath] = useState<MediaItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [userId, setUserId] = useState<string | null>(null);

    const getJellyfinUserId = useCallback(async () => {
        if (!settings.mediaServerUrl || !settings.mediaServerApiKey) {
            setError('Media server URL or API key is not configured.');
            return;
        }
        try {
            const response = await fetch(`${settings.mediaServerUrl}/Users`, {
                headers: {
                    'X-Emby-Token': settings.mediaServerApiKey,
                },
            });
            if (!response.ok) {
                throw new Error(`Failed to fetch users: ${response.statusText}`);
            }
            const users = await response.json();
            if (users && users.length > 0) {
                setUserId(users[0].Id);
            } else {
                throw new Error('No users found on the Jellyfin server.');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        }
    }, [settings.mediaServerUrl, settings.mediaServerApiKey]);

    const fetchFromJellyfin = useCallback(async (endpoint: string, params: Record<string, string> = {}) => {
        if (!settings.mediaServerUrl || !settings.mediaServerApiKey || !userId) {
            setError('Media server URL, API key, or User ID is not available.');
            return null;
        }
        setLoading(true);
        setError(null);
        try {
            const url = new URL(`${settings.mediaServerUrl}/Users/${userId}/${endpoint}`);
            Object.entries(params).forEach(([key, value]) => url.searchParams.append(key, value));

            const response = await fetch(url.toString(), {
                headers: {
                    'X-Emby-Token': settings.mediaServerApiKey,
                },
            });
            if (!response.ok) {
                throw new Error(`Failed to fetch from Jellyfin: ${response.statusText}`);
            }
            const data = await response.json();
            return data.Items || data;
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
            return null;
        } finally {
            setLoading(false);
        }
    }, [settings.mediaServerUrl, settings.mediaServerApiKey, userId]);

    const loadMediaItems = useCallback(async () => {
        if (!userId) return;

        if (query) {
            const items = await fetchFromJellyfin('Items', { 
                searchTerm: query, 
                IncludeItemTypes: 'Movie,Series' 
            });
            if (items) {
                setMediaItems(items);
            }
        } else if (currentPath.length > 0) {
            const parent = currentPath[currentPath.length - 1];
            const items = await fetchFromJellyfin('Items', { ParentId: parent.Id });
            if (items) {
                setMediaItems(items);
            }
        } else {
            const views = await fetchFromJellyfin('Views');
            if (views) {
                setMediaItems(views);
            }
        }
    }, [query, currentPath, fetchFromJellyfin, userId]);

    useEffect(() => {
        getJellyfinUserId();
    }, [getJellyfinUserId]);

    useEffect(() => {
        if (userId) {
            loadMediaItems();
        }
    }, [userId, loadMediaItems]);

    const navigateToItem = (item: MediaItem) => {
        if (item.IsFolder) {
            setCurrentPath([...currentPath, item]);
        } else {
            onPlay(item.Id, item.Type);
        }
    };

    const navigateBack = () => {
        if (currentPath.length > 0) {
            const newPath = currentPath.slice(0, -1);
            setCurrentPath(newPath);
        }
    };

    const getImageUrl = (item: MediaItem) => {
        if (!settings.mediaServerUrl) return '';
        return `${settings.mediaServerUrl}/Items/${item.Id}/Images/Primary`;
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-4xl max-h-[80vh] flex flex-col">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-2xl font-bold text-white">Media Browser</h3>
                    {currentPath.length > 0 && (
                        <button onClick={navigateBack} className="px-4 py-2 rounded bg-gray-600 text-white hover:bg-gray-500 transition-colors">
                            ← Back
                        </button>
                    )}
                </div>

                {loading && <div className="text-white text-center p-8">Loading...</div>}
                {error && <div className="text-red-500 text-center p-8">{error}</div>}

                {!loading && !error && (
                    <div className="overflow-y-auto flex-grow">
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                            {mediaItems.map(item => (
                                <div
                                    key={item.Id}
                                    className="bg-gray-700 rounded-lg overflow-hidden cursor-pointer transform hover:scale-105 transition-transform duration-200"
                                    onClick={() => navigateToItem(item)}
                                >
                                    <img src={getImageUrl(item)} alt={item.Name} className="w-full h-48 object-cover" />
                                    <div className="p-2">
                                        <h4 className="text-white font-bold truncate">{item.Name}</h4>
                                        {item.Year && <span className="text-gray-400 text-sm">{item.Year}</span>}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div className="flex justify-end mt-4">
                    <button onClick={onClose} className="px-4 py-2 rounded bg-gray-600 text-white hover:bg-gray-500 transition-colors">
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default MediaBrowser;
