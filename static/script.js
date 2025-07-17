// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    // Get references to DOM elements
    const contentImageInput = document.getElementById('content_image');
    const styleImageInput = document.getElementById('style_image');
    const contentPreview = document.getElementById('content_preview');
    const stylePreview = document.getElementById('style_preview');

    const outputSizeSelect = document.getElementById('output_size');
    const numEpochsInput = document.getElementById('num_epochs');
    const learningRateInput = document.getElementById('learning_rate');
    const alphaInput = document.getElementById('alpha');
    const betaInput = document.getElementById('beta');
    
    const suggestStylesButton = document.getElementById('suggest_styles_button');
    const suggestedStylesContainer = document.getElementById('suggested_styles_container');
    const stylizeButton = document.getElementById('stylize_button');

    const processingMessage = document.getElementById('processing_message');
    const errorMessageElement = document.getElementById('error_message');
    
    const resultsArea = document.getElementById('results_area');
    const resultImage = document.getElementById('result_image');
    const downloadLink = document.getElementById('download_link');
    
    const intermediateResultsArea = document.getElementById('intermediate_results_area');
    const intermediateImagesContainer = document.getElementById('intermediate_images_container');

    // --- Function to preview uploaded images ---
    function previewImage(inputElement, previewElement) {
        inputElement.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewElement.src = e.target.result;
                    previewElement.style.display = 'block';
                }
                reader.readAsDataURL(this.files[0]);
            } else {
                previewElement.src = '#';
                previewElement.style.display = 'none';
            }
        });
    }
    previewImage(contentImageInput, contentPreview);
    previewImage(styleImageInput, stylePreview);


    // --- Event Listener for Suggest Styles Button ---
    suggestStylesButton.addEventListener('click', async () => {
        if (!contentImageInput.files[0]) {
            alert('Please select a content image first to get style suggestions.');
            return;
        }

        suggestStylesButton.disabled = true;
        suggestStylesButton.textContent = 'Suggesting...';
        suggestedStylesContainer.innerHTML = '<p>Loading suggestions...</p>';
        errorMessageElement.style.display = 'none'; // Clear previous errors

        const formData = new FormData();
        formData.append('content_image', contentImageInput.files[0]);

        try {
            const response = await fetch('/suggest_styles', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            suggestedStylesContainer.innerHTML = ''; // Clear "Loading suggestions..."

            if (response.ok && data.suggested_styles) {
                if (data.suggested_styles.length > 0) {
                    data.suggested_styles.forEach(styleUrl => {
                        const imgThumbnail = document.createElement('img');
                        imgThumbnail.src = `/static/${styleUrl}`;
                        imgThumbnail.title = `Use ${styleUrl.split('/').pop()} as style`;

                        imgThumbnail.addEventListener('click', async () => {
                            document.querySelectorAll('#suggested_styles_container img').forEach(img => img.classList.remove('selected-style'));
                            imgThumbnail.classList.add('selected-style');
                            
                            // Show a small loading indicator for style selection
                            const tempProcessing = document.createElement('p');
                            tempProcessing.textContent = "Loading selected style...";
                            suggestedStylesContainer.appendChild(tempProcessing);

                            try {
                                const styleResponse = await fetch(imgThumbnail.src);
                                const styleBlob = await styleResponse.blob();
                                const styleFile = new File([styleBlob], styleUrl.split('/').pop(), { type: styleBlob.type });

                                const dataTransfer = new DataTransfer();
                                dataTransfer.items.add(styleFile);
                                styleImageInput.files = dataTransfer.files;
                                
                                // Trigger change event for preview on styleImageInput
                                const event = new Event('change');
                                styleImageInput.dispatchEvent(event);
                                
                                console.log(`Selected style via suggestion: ${styleFile.name}`);
                            } catch (e) {
                                console.error("Error loading selected style from suggestion:", e);
                                alert("Could not load the selected style image.");
                            } finally {
                                if (suggestedStylesContainer.contains(tempProcessing)) {
                                    suggestedStylesContainer.removeChild(tempProcessing);
                                }
                            }
                        });
                        suggestedStylesContainer.appendChild(imgThumbnail);
                    });
                } else {
                    suggestedStylesContainer.innerHTML = '<p>No style suggestions found for this content image.</p>';
                }
            } else {
                suggestedStylesContainer.innerHTML = `<p style="color:red;">Error suggesting styles: ${data.error || 'Unknown server error'}</p>`;
            }
        } catch (error) {
            console.error('Error fetching style suggestions:', error);
            suggestedStylesContainer.innerHTML = '<p style="color:red;">Could not fetch style suggestions. Network or server issue.</p>';
        } finally {
            suggestStylesButton.disabled = false;
            suggestStylesButton.textContent = 'Suggest Styles';
        }
    });


    // --- Event Listener for Main Stylize Button ---
    stylizeButton.addEventListener('click', async () => {
        // Clear previous results and errors
        errorMessageElement.textContent = '';
        errorMessageElement.style.display = 'none';
        resultsArea.style.display = 'none';
        resultImage.src = '#';
        downloadLink.style.display = 'none';
        downloadLink.href = '#';
        intermediateResultsArea.style.display = 'none';
        intermediateImagesContainer.innerHTML = '';

        if (!contentImageInput.files[0]) {
            errorMessageElement.textContent = 'Please select a content image.';
            errorMessageElement.style.display = 'block';
            return;
        }
        if (!styleImageInput.files[0]) {
            errorMessageElement.textContent = 'Please select a style image.';
            errorMessageElement.style.display = 'block';
            return;
        }

        processingMessage.style.display = 'block';
        stylizeButton.disabled = true;
        stylizeButton.textContent = 'Stylizing...';

        const formData = new FormData();
        formData.append('content_image', contentImageInput.files[0]);
        formData.append('style_image', styleImageInput.files[0]);
        formData.append('output_size', outputSizeSelect.value);
        formData.append('num_epochs', numEpochsInput.value);
        formData.append('learning_rate', learningRateInput.value);
        formData.append('alpha', alphaInput.value);
        formData.append('beta', betaInput.value);

        try {
            const response = await fetch('/stylize', {
                method: 'POST',
                body: formData,
            });

            let data;
            try {
                data = await response.json();
            } catch (jsonError) {
                console.error('Failed to parse JSON response from /stylize:', jsonError);
                const textResponse = await response.text();
                console.error('Server response text from /stylize:', textResponse);
                errorMessageElement.textContent = `Error: Server returned non-JSON response during stylization. Status: ${response.status}. Check console.`;
                errorMessageElement.style.display = 'block';
                return; // Exit early
            } finally { // Always hide processing and re-enable button in this try's finally
                processingMessage.style.display = 'none';
                stylizeButton.disabled = false;
                stylizeButton.textContent = 'Stylize Image!';
            }

            if (response.ok && data.result_image_url) {
                console.log("Stylization Success data from server:", data);
                resultsArea.style.display = 'block';
                resultImage.src = `/static/${data.result_image_url}`;
                resultImage.style.display = 'block';

                downloadLink.href = `/static/${data.result_image_url}`;
                const filename = data.result_image_url.substring(data.result_image_url.lastIndexOf('/') + 1);
                downloadLink.setAttribute('download', filename || 'stylized_image.png');
                downloadLink.style.display = 'inline-block';

                if (data.intermediate_image_urls && data.intermediate_image_urls.length > 0) {
                    intermediateResultsArea.style.display = 'block';
                    data.intermediate_image_urls.forEach(url => {
                        const imgElement = document.createElement('img');
                        imgElement.src = `/static/${url}`;
                        imgElement.alt = 'Intermediate Step';
                        imgElement.addEventListener('click', () => window.open(`/static/${url}`, '_blank')); // Open in new tab on click
                        intermediateImagesContainer.appendChild(imgElement);
                    });
                }
            } else {
                console.error("Stylization Error data from server:", data);
                errorMessageElement.textContent = `Server Error: ${data.error || response.statusText || 'Unknown error during stylization.'}`;
                errorMessageElement.style.display = 'block';
            }

        } catch (error) { // Network error for /stylize
            console.error('Fetch error or server issue for /stylize:', error);
            processingMessage.style.display = 'none';
            stylizeButton.disabled = false;
            stylizeButton.textContent = 'Stylize Image!';
            errorMessageElement.textContent = `Client-side Error during stylization: ${error.message}. Check console. Is the Flask server running?`;
            errorMessageElement.style.display = 'block';
        }
    });
});