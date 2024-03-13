class NsfwDetector {
    constructor() {
        this._threshold = 0.35;
        this._nsfwLabels = [
            'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED', 'BLOOD_SHED', 'VIOLENCE', 'GORE', 'PORNOGRAPHY', 'DRUGS', 'ALCOHOL',
        ];
        // Load the TensorFlow model once and reuse
        this._classifierPromise = window.tensorflowPipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
    }

    async isNsfw(imageUrl) {
        let blobUrl = '';
        try {
            blobUrl = await this._loadAndResizeImage(imageUrl);
            const classifier = await this._classifierPromise; // Use the preloaded model
            const output = await classifier(blobUrl, this._nsfwLabels);
            console.log(output);
            const nsfwDetected = output.some(result => result.score > this._threshold);
            return nsfwDetected;
        } catch (error) {
            console.error('Error during NSFW classification: ', error);
            throw error;
        } finally {
            if (blobUrl) {
                URL.revokeObjectURL(blobUrl); // Free up memory
            }
        }
    }

    async _loadAndResizeImage(imageUrl) {
        const img = await this._loadImage(imageUrl);
        const offScreenCanvas = document.createElement('canvas');
        const ctx = offScreenCanvas.getContext('2d');
        offScreenCanvas.width = 224;
        offScreenCanvas.height = 224;
    
        ctx.drawImage(img, 0, 0, offScreenCanvas.width, offScreenCanvas.height);
        
        return new Promise((resolve, reject) => {
            offScreenCanvas.toBlob(blob => {
                if (!blob) {
                    reject('Canvas to Blob conversion failed');
                    return;
                }
                const blobUrl = URL.createObjectURL(blob);
                resolve(blobUrl);
            }, 'image/jpeg');
        });
    }

    async _loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = () => reject(`Failed to load image: ${url}`);
            img.src = url;
        });
    }
}

window.NsfwDetector = NsfwDetector;



window.NsfwDetector = NsfwDetector;



