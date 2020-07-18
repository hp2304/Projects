package com.resumeBuilder.controller.company;

import com.resumeBuilder.model.company.Company;
import com.resumeBuilder.model.company.CompanyStorage;
import com.resumeBuilder.view.MainView;
import com.resumeBuilder.view.company.CompanyLogin;
import com.resumeBuilder.view.company.CompanyOptionsView;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

public class CompanyLoginView implements ActionListener {

    private CompanyLogin comLoginView;

    public CompanyLoginView(CompanyLogin comLoginView) {
        this.comLoginView = comLoginView;
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        try {
            if (e.getActionCommand().equals("<--")) {
                MainView mainView = new MainView();
                mainView.setVisible(true);
                comLoginView.setVisible(false);
            } else {
                ArrayList<Company> company = new ArrayList<>();
                company = CompanyStorage.readCompanies();
                if (company != null) {
                    Company obj = FindCompany.findCompany(company, comLoginView.getTxtCUserName(), comLoginView.getPasswordField());
                    if (obj == null) {
                        JOptionPane.showMessageDialog(null, "User Name or Password Incorrect");
                    } else {
                        JOptionPane.showMessageDialog(null, "LogIn Successfully...");
                        CompanyOptionsView frame = new CompanyOptionsView(obj);
                        frame.setVisible(true);
                        comLoginView.setVisible(false);
                    }
                } else {
                    JOptionPane.showMessageDialog(null, "File not Found");
                }
            }
        } catch (Exception err) {
            err.printStackTrace();
        }
    }
}
