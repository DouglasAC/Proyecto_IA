
var departamentos = []

function Alcargar() {


    $.ajax({
        async: false,
        contentType: 'application/json;  charset=utf-8',
        type: "POST",
        url: "/datos",
        data: JSON.stringify({
        }),
        success: function (data, textStatus, jqXHR) {
            console.log(data)
            valores = data.depa
            departamentos = valores
            let dep = ""
            for (let x = 0; x < departamentos.length; x++) {
                dep += "<option>" + departamentos[x][0] + "</option>\n"
            }
            document.getElementById("departamento").innerHTML = dep;
            let mun = ""
            let munis = departamentos[0][1]
            for (let y = 0; y < munis.length; y++) {
                mun += "<option value = " + munis[y][1] + ">" + munis[y][0] + "</option>\n"
            }
            document.getElementById("municipio").innerHTML = mun;
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.log(textStatus)
        }

    });



}

function cambiarMunicipios()
{
    let dep =  document.getElementById("departamento").value
    console.log(dep)
    let depss =  document.getElementById("municipio").value
    console.log(depss)
    for (let x = 0; x < departamentos.length; x++) {
        if(departamentos[x][0] == dep)
        {
            let mun = ""
            let munis = departamentos[x][1]
            for (let y = 0; y < munis.length; y++) {
                mun += "<option value = " + munis[y][1] + ">" + munis[y][0] + "</option>\n"
            }
            document.getElementById("municipio").innerHTML = mun;
            break
        }
    }
    let deps =  document.getElementById("municipio").value
    console.log(deps)
}


function enviar()
{
    let genero = document.getElementById("genero").value
    let edad = document.getElementById("edad").value
    let departamento = document.getElementById("departamento").value
    let municipio = document.getElementById("municipio").value
    let year = document.getElementById("year").value

    console.log(genero)
    console.log(edad)
    console.log(year)
    console.log(departamento)
    console.log(municipio)
    $.ajax({
        async: false,
        contentType: 'application/json;  charset=utf-8',
        type: "POST",
        url: "/predecir",
        data: JSON.stringify({
            genero: genero,
            edad: edad,
            year: year,
            distancia: municipio
        }),
        success: function (data, textStatus, jqXHR) {
            console.log(data)
            
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.log(textStatus)
        }

    });
}