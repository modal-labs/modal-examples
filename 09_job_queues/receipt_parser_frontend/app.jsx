function Form() {
  const [selectedFile, setSelectedFile] = React.useState();

  const changeHandler = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmission = async () => {
    const formData = new FormData();
    formData.append("receipt", selectedFile);

    const resp = await fetch("/parse", {
      method: "POST",
      body: formData,
    });

    if (resp.status !== 200) {
      throw new Error("An error occurred: " + resp.status);
    }
    console.log(await resp.json());
  };

  return (
    <form class="flex flex-col space-y-2">
      <div> Upload your receipt: </div>
      <input type="file" name="file" onChange={changeHandler} />
      <div>
        <button type="button" onClick={handleSubmission}>Submit</button>
      </div>
    </form>
  );
}

function App() {
  return (
    <div class="absolute inset-0 bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300">
      <div class="mx-auto max-w-md py-8">
        <main class="rounded-xl bg-white p-6">
          <Form />
        </main>
      </div>
    </div>
  );
}


const container = document.getElementById('react');
ReactDOM.createRoot(container).render(<App />);