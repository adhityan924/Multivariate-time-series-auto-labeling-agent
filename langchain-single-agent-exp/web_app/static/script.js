document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const analysisForm = document.getElementById('analysisForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const results = document.getElementById('results');

    // Handle file upload
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const fileInput = document.getElementById('file');
        formData.append('file', fileInput.files[0]);

        try {
            uploadStatus.innerHTML = '<div class="alert alert-info fade-in">Uploading and processing file...</div>';
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                uploadStatus.innerHTML = '<div class="alert alert-success fade-in">' + data.message + '</div>';
            } else {
                uploadStatus.innerHTML = '<div class="alert alert-danger fade-in">Error: ' + data.message + '</div>';
            }
        } catch (error) {
            uploadStatus.innerHTML = '<div class="alert alert-danger fade-in">Error: ' + error.message + '</div>';
        }
    });

    // Handle pattern analysis
    analysisForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const pattern = document.getElementById('pattern').value;
        const label = document.getElementById('label').value;
        
        const formData = new FormData();
        formData.append('pattern', pattern);
        formData.append('label', label);

        try {
            results.innerHTML = '<p class="text-info fade-in">Analyzing pattern...</p>';
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Format the results
                let resultsHtml = '<div class="fade-in">';
                resultsHtml += '<h6 class="mb-3">Similar Patterns Found:</h6>';
                
                if (data.results && data.results.length > 0) {
                    resultsHtml += '<div class="table-responsive"><table class="table table-hover">';
                    resultsHtml += '<thead><tr><th>Pattern</th><th>Similarity Score</th></tr></thead>';
                    resultsHtml += '<tbody>';
                    
                    data.results.forEach((result, index) => {
                        resultsHtml += `<tr>
                            <td>${JSON.stringify(result)}</td>
                            <td>${(1 - index * 0.1).toFixed(2)}</td>
                        </tr>`;
                    });
                    
                    resultsHtml += '</tbody></table></div>';
                } else {
                    resultsHtml += '<p class="text-muted">No similar patterns found.</p>';
                }
                
                resultsHtml += '</div>';
                results.innerHTML = resultsHtml;
            } else {
                results.innerHTML = '<div class="alert alert-danger fade-in">Error: ' + data.message + '</div>';
            }
        } catch (error) {
            results.innerHTML = '<div class="alert alert-danger fade-in">Error: ' + error.message + '</div>';
        }
    });
}); 